"""Tests for sensaug/round_schedule.py -- the SA val-round grid and the
correlation pipeline's firing schedule.

Deliberately free of mmseg, torch and the registries, like tests/test_redundancy.py:
this is integer arithmetic that decides where every gradient sweep in a run lands,
and getting it wrong produces a run that looks completely normal and measures the
wrong thing. It must be checkable without a GPU.

The anchor is the round table this schedule was specified from -- the pure
SA_CURVE_CADENCE-only mechanism (`derive_corr_rounds(20, 4)`, no cadence override):

    rounds 0-2   nothing
    round 3      one control probe (baseline, does not feed the pdf)
    round 4      compute, used for rounds 4-9
    round 10     compute, used for rounds 10-15
    round 16     compute, used for rounds 16-18
    round 19     nothing (training's over)

The CHECKED-IN configs/rounds.yaml sets `corr_cadence: 3`, which is denser than
that mechanism default -- it derives [3, 4, 7, 10, 13, 16] instead. Rounds 7 and
13 land inside an already-current SA-curve window (4-9, 10-15) and refresh
`red(a)` without moving the curve itself, which still only recomputes every
SA_CURVE_CADENCE rounds. See test_the_checked_in_config_produces_the_table.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.round_schedule import (
    DEFAULT_ROUNDS_CONFIG,
    SA_CURVE_CADENCE,
    RoundConfig,
    control_rounds,
    derive_corr_rounds,
    load_round_config,
    resolve_schedule,
    round_fire_iters,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(**kwargs):
    base = dict(n_rounds=20, warmup_rounds=4, path="<test>")
    base.update(kwargs)
    return RoundConfig(**base)


# ---------------------------------------------------------------- the table


def test_the_default_schedule_is_the_specified_round_table():
    """3, 4, 10, 16 -- the baseline probe plus one emission per SA curve. This is
    the whole specification; if it ever changes, it should change here first."""
    assert derive_corr_rounds(20, 4) == [3, 4, 10, 16]


def test_the_firing_rounds_are_exactly_the_sa_curve_recompute_rounds():
    """RobustValLoop.run recomputes the curve when (n_rounds - warmup) % 6 == 0.
    Every non-control emission must land on one of those, or it is measuring a
    model whose pdf shape is about to be governed by a curve it never saw."""
    rounds = derive_corr_rounds(20, 4)
    for r in rounds:
        if r in control_rounds(rounds, 4):
            continue
        assert (r - 4) % SA_CURVE_CADENCE == 0


def test_the_control_probe_is_the_last_warmup_round():
    rounds = derive_corr_rounds(20, 4)
    assert control_rounds(rounds, 4) == [3]


def test_the_final_round_never_fires():
    """Training ends with it, so nothing could ever read the R it emitted."""
    for n_rounds in range(2, 60):
        for warmup in range(0, min(n_rounds, 8)):
            assert n_rounds - 1 not in derive_corr_rounds(n_rounds, warmup)


def test_no_warmup_has_no_control_probe():
    """With warmup_rounds=0 training is pdf-driven from round 0, so no round of
    that run is a clean pre-pdf baseline and claiming one would be a lie."""
    rounds = derive_corr_rounds(20, 0)
    assert rounds == [0, 6, 12, 18]
    assert control_rounds(rounds, 0) == []


def test_the_count_follows_the_round_grid_rather_than_being_pinned_at_four():
    """4 emissions is what 20 rounds happens to give, not a target. Halving
    round_interval doubles the rounds and should buy more matrices, not the same
    four spread further apart."""
    assert len(derive_corr_rounds(20, 4)) == 4
    assert len(derive_corr_rounds(40, 4)) == 7


# ------------------------------------------------- rounds -> iterations


def test_round_r_fires_at_the_hook_point_just_before_round_r_val_loop():
    """The load-bearing off-by-one. IterBasedTrainLoop.run_iter calls
    after_train_iter and only THEN increments _iter and tests `_iter %
    val_interval`, so the hook point preceding round r's val loop is the one where
    iteration_count() == (r + 1) * round_interval.

    Asserted against a replay of that loop rather than against the formula, so an
    inverted mapping cannot pass by matching itself.
    """
    round_interval = 4000
    max_iters = 80000
    fire = set(round_fire_iters(derive_corr_rounds(20, 4), round_interval))

    # Replay mmengine's loop: hook point at iteration_count, then the val test.
    hook_points_before_a_val_round = {}
    n_rounds_seen = 0
    for zero_based_iter in range(max_iters):
        iteration_count = zero_based_iter + 1  # what fires_at() computes
        # ... after_train_iter runs here ...
        if iteration_count % round_interval == 0:  # ... then this
            hook_points_before_a_val_round[n_rounds_seen] = iteration_count
            n_rounds_seen += 1

    assert fire == {hook_points_before_a_val_round[r] for r in (3, 4, 10, 16)}
    assert hook_points_before_a_val_round[19] == max_iters  # and it does NOT fire
    assert max_iters not in fire


def test_round_fire_iters_at_the_default_geometry():
    assert round_fire_iters([3, 4, 10, 16], 4000) == (16000, 20000, 44000, 68000)


def test_init_sa_shifts_every_round_one_interval_earlier():
    """RobustIterBasedTrainLoop runs one val round BEFORE the training loop when
    init_sa is set (--resume, --no-warmup). That round consumes round number 0 at
    iteration 0, so every later round arrives an interval sooner."""
    assert round_fire_iters([3, 4, 10, 16], 4000, pre_train_round=True) == (
        12000,
        16000,
        40000,
        64000,
    )


def test_a_round_with_no_hook_point_is_dropped_not_emitted_at_zero():
    """Under init_sa, round 0 happens before any training iteration, so there is
    no after_train_iter at which it could fire. Emitting it at iteration 0 would
    put a sweep before the model has taken a single step."""
    assert round_fire_iters([0, 6], 4000, pre_train_round=True) == (24000,)


# ----------------------------------------------------------- the config file


def test_the_checked_in_config_produces_the_table():
    """configs/rounds.yaml ships corr_cadence: 3, denser than the SA_CURVE_CADENCE
    (6) mechanism default -- this is what makes a real run fire more than the 4
    times the pure-mechanism table above would give."""
    cfg = load_round_config(os.path.join(REPO, DEFAULT_ROUNDS_CONFIG))
    assert (cfg.n_rounds, cfg.warmup_rounds, cfg.corr_cadence) == (20, 4, 3)
    assert cfg.corr_rounds is None and cfg.control_rounds is None

    schedule = resolve_schedule(cfg, 20, 4, 4000)
    assert schedule.rounds == (3, 4, 7, 10, 13, 16)
    assert schedule.fire_iters == (16000, 20000, 32000, 44000, 56000, 68000)
    assert schedule.control_iters == (16000,)
    assert schedule.derived


def test_a_missing_file_raises_rather_than_falling_back_to_defaults(tmp_path):
    """A silent fallback would produce a plausible-looking experiment measuring
    something other than what the launch command says."""
    with pytest.raises(FileNotFoundError, match="rounds.yaml"):
        load_round_config(str(tmp_path / "nope.yaml"))


def test_an_unknown_key_raises(tmp_path):
    """A typo'd key that silently does nothing is how a schedule change gets
    written up as 'had no effect'."""
    path = tmp_path / "rounds.yaml"
    path.write_text(yaml.safe_dump({"n_rounds": 20, "warmup_rounds": 4, "corr_round": [3]}))
    with pytest.raises(ValueError, match="unknown key"):
        load_round_config(str(path))


@pytest.mark.parametrize("key", ["n_rounds", "warmup_rounds"])
def test_the_required_keys_may_not_be_null(tmp_path, key):
    body = {"n_rounds": 20, "warmup_rounds": 4}
    body[key] = None
    path = tmp_path / "rounds.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match=key):
        load_round_config(str(path))


def test_a_non_integer_round_raises(tmp_path):
    path = tmp_path / "rounds.yaml"
    path.write_text(
        yaml.safe_dump({"n_rounds": 20, "warmup_rounds": 4, "corr_rounds": [3, "four"]})
    )
    with pytest.raises(ValueError, match="non-integer"):
        load_round_config(str(path))


def test_corr_cadence_parses_from_yaml(tmp_path):
    path = tmp_path / "rounds.yaml"
    path.write_text(yaml.safe_dump({"n_rounds": 20, "warmup_rounds": 4, "corr_cadence": 2}))
    cfg = load_round_config(str(path))
    assert cfg.corr_cadence == 2


def test_corr_cadence_defaults_to_none_when_omitted(tmp_path):
    path = tmp_path / "rounds.yaml"
    path.write_text(yaml.safe_dump({"n_rounds": 20, "warmup_rounds": 4}))
    cfg = load_round_config(str(path))
    assert cfg.corr_cadence is None


@pytest.mark.parametrize("bad", [0, -1, "two", True])
def test_corr_cadence_must_be_a_positive_int(tmp_path, bad):
    path = tmp_path / "rounds.yaml"
    path.write_text(
        yaml.safe_dump({"n_rounds": 20, "warmup_rounds": 4, "corr_cadence": bad})
    )
    with pytest.raises(ValueError, match="corr_cadence"):
        load_round_config(str(path))


# ------------------------------------------------------------- resolution


def test_explicit_rounds_win_over_the_derived_ones():
    schedule = resolve_schedule(
        _cfg(corr_rounds=[3, 7, 11], control_rounds=[3]), 20, 4, 4000
    )
    assert schedule.rounds == (3, 7, 11)
    assert schedule.fire_iters == (16000, 32000, 48000)
    assert schedule.control_iters == (16000,)
    assert not schedule.derived


def test_explicit_rounds_still_derive_their_controls_when_left_null():
    schedule = resolve_schedule(_cfg(corr_rounds=[2, 3, 9]), 20, 4, 4000)
    assert schedule.control_iters == (12000, 16000)  # both fall inside warmup


def test_an_out_of_grid_round_raises():
    with pytest.raises(ValueError, match=r"round\(s\) \[25\]"):
        resolve_schedule(_cfg(corr_rounds=[3, 25]), 20, 4, 4000)
    with pytest.raises(ValueError, match=r"round\(s\) \[-1\]"):
        resolve_schedule(_cfg(corr_rounds=[-1, 3]), 20, 4, 4000)


def test_a_final_round_entry_warns_but_is_kept():
    """An explicit list is a deliberate statement, so it is honoured -- but the R
    it emits is measured after the last pdf is built and nothing can read it."""
    schedule = resolve_schedule(_cfg(corr_rounds=[4, 19], control_rounds=[]), 20, 4, 4000)
    assert 19 in schedule.rounds
    assert any("final round" in w for w in schedule.warnings)


def test_a_control_round_that_never_fires_raises():
    """A baseline that is silently absent is worse than one that is explicitly
    disabled -- nothing downstream would reveal it."""
    with pytest.raises(ValueError, match="control_rounds"):
        resolve_schedule(_cfg(corr_rounds=[4, 10], control_rounds=[3]), 20, 4, 4000)


def test_an_empty_explicit_list_is_refused():
    """`[]` is not the same as null: null derives a schedule, `[]` asks for a
    registered pipeline that can never emit."""
    with pytest.raises(ValueError, match="fires nowhere"):
        resolve_schedule(_cfg(corr_rounds=[]), 20, 4, 4000)


def test_too_few_rounds_names_both_ways_out():
    """A one-round run has no round the grid can use: the only round is the final
    one, and the final round never fires."""
    with pytest.raises(ValueError, match="--round_interval.*--corr-interval"):
        resolve_schedule(_cfg(), n_rounds=1, warmup_rounds=0, round_interval=80000)


def test_warmup_that_swallows_the_run_raises():
    with pytest.raises(ValueError, match="no post-warmup round"):
        resolve_schedule(_cfg(), n_rounds=4, warmup_rounds=4, round_interval=4000)


def test_resolve_schedule_uses_cfg_corr_cadence_when_set():
    schedule = resolve_schedule(_cfg(corr_cadence=2), 20, 4, 4000)
    assert schedule.rounds == tuple(derive_corr_rounds(20, 4, cadence=2))
    assert schedule.rounds == (3, 4, 6, 8, 10, 12, 14, 16, 18)


def test_an_explicit_cadence_argument_overrides_cfg_corr_cadence():
    schedule = resolve_schedule(_cfg(corr_cadence=2), 20, 4, 4000, cadence=6)
    assert schedule.rounds == (3, 4, 10, 16)


def test_the_schedule_follows_the_effective_round_count_not_the_configured_one():
    """rounds.yaml's n_rounds is only the default divisor for round_interval.
    --round_interval=2000 on an 80k run gives 40 rounds, and the schedule must
    describe those 40 -- deriving it from the configured 20 would point every
    round at the wrong iteration while still looking right in the log."""
    cfg = _cfg(n_rounds=20)
    schedule = resolve_schedule(cfg, n_rounds=40, warmup_rounds=4, round_interval=2000)
    assert schedule.rounds == (3, 4, 10, 16, 22, 28, 34)
    assert schedule.fire_iters[0] == 8000  # (3 + 1) * 2000
