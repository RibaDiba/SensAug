"""Tests for the augmentation gradient cross-correlation matrix R.

The maths here is pure numpy, but the module imports the mmseg/mmengine
registries, so run these in the `sensaug` conda env on a compute node.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mmengine")
pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The merged 32-op vocabulary -- the hook's `self.names`, and therefore the axes
# of R. Fixtures here are keyed by op name, so they must cover the same set the
# hook iterates or stack_probe_buffer raises KeyError on the missing ones.
from sensaug.dataset.differentiable_augmentations_aa import (
    ALL_DIFFERENTIABLE_PERTURBATIONS as DIFFERENTIABLE_PERTURBATIONS,
)

from sensaug.hooks.grad_sens_analysis import (
    PerturbationSensitivityAnalysisHookWithGradients,
    _jsonable,
    closure_null,
    cluster_bootstrap_ci,
    correlate,
    fdr_correct,
    merge_rank_buffers,
    shared_factor_loadings,
    shared_image_factor,
    stack_probe_buffer,
)

NAMES = list(DIFFERENTIABLE_PERTURBATIONS)
N_OPS = len(NAMES)


def _buffer(probes_per_op):
    """probes_per_op: list of (B,) arrays, reused for every op."""
    return {name: [p.copy() for p in probes_per_op] for name in NAMES}


def _confounded_grid(n_ops=N_OPS, n_images=800, seed=0):
    """Ops that are genuinely INDEPENDENT of one another, observed through a
    shared per-image scale.

    Two facts drive this: augmentation sensitivities are predominantly positive
    (perturbing an image almost always increases its loss), and a hard image has a
    big gradient for EVERY op. So a shared per-image factor multiplies rows that
    have a positive mean -- which manufactures correlation out of nothing.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=1.0, scale=0.15, size=(n_ops, n_images))
    difficulty = rng.uniform(0.5, 5.0, size=n_images)
    return base * difficulty


# --- building D over images --------------------------------------------------


def test_stack_probe_buffer_concatenates_over_images_not_probes():
    """The axis IS the fix. 5 probes of 4 images must give 20 columns to correlate
    over, not 5."""
    probes = [np.arange(4, dtype=float) + i for i in range(5)]
    d_grad, probe_ids = stack_probe_buffer(_buffer(probes), NAMES)

    assert d_grad.shape == (N_OPS, 20)
    assert probe_ids.shape == (20,)
    assert np.unique(probe_ids).size == 5
    # column j carries the probe it came from -- the bootstrap's cluster id
    np.testing.assert_array_equal(probe_ids[:4], [0, 0, 0, 0])
    np.testing.assert_array_equal(probe_ids[4:8], [1, 1, 1, 1])


def test_stack_probe_buffer_allows_ragged_batches():
    """A short final batch (drop_last=False) shortens every op's row equally, so
    the columns stay aligned."""
    probes = [np.zeros(4), np.zeros(4), np.zeros(2)]
    d_grad, probe_ids = stack_probe_buffer(_buffer(probes), NAMES)

    assert d_grad.shape == (N_OPS, 10)
    assert np.unique(probe_ids).size == 3


def test_stack_probe_buffer_rejects_misaligned_ops():
    """Column j must mean the same image for every op. If one op is missing a
    probe, correlating the rows silently compares different images -- so this is
    an assert, not a truncation."""
    buffer = _buffer([np.zeros(4), np.zeros(4)])
    buffer[NAMES[3]] = [np.zeros(4)]  # one op fell behind

    with pytest.raises(AssertionError, match="misaligned"):
        stack_probe_buffer(buffer, NAMES)


# --- correlation + guards ----------------------------------------------------


def test_correlate_recovers_identical_and_independent_rows():
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 500))
    d_grad[1] = d_grad[0] + 1e-6 * rng.normal(size=500)  # a redundant pair

    r, dropped = correlate(d_grad)

    assert dropped.size == 0
    assert r[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert r[2, 3] == pytest.approx(0.0, abs=0.15)


def test_correlate_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    r, _ = correlate(rng.normal(size=(N_OPS, 300)))

    assert r.shape == (N_OPS, N_OPS)
    np.testing.assert_allclose(r, r.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(r), np.ones(N_OPS), atol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_correlate_drops_a_constant_row_instead_of_emitting_a_silent_nan():
    """An op with no variance over the window has no correlation signal at all.
    np.corrcoef would divide by its zero std, emit a RuntimeWarning, and quietly
    scatter NaN through the matrix. Drop it explicitly instead, and make sure the
    kept submatrix comes back clean (filterwarnings turns the warning into a
    failure)."""
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 200))
    d_grad[2] = 0.7  # this op never moves

    r, dropped = correlate(d_grad)

    np.testing.assert_array_equal(dropped, [2])
    assert np.isnan(r[2]).all()  # whole row, diagonal included
    assert np.isnan(r[:, 2]).all()
    keep = [0, 1, 3]
    assert np.isfinite(r[np.ix_(keep, keep)]).all()


# --- the image-difficulty confound -------------------------------------------


def test_shared_image_scale_inflates_raw_r_and_normalization_removes_it():
    """The failure mode this whole normalization exists for.

    Correlating raw per-image gradients does not measure redundancy -- it measures
    image difficulty. These 14 ops are independent by construction, yet raw R
    calls almost every pair redundant, for exactly the same reason the old
    correlate-over-training-time R did: a factor shared by every row.
    """
    d_grad = _confounded_grid()
    off_diagonal = np.triu_indices(N_OPS, k=1)

    r_raw, _ = correlate(d_grad)
    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    # Raw R: "everything is redundant." Nothing is.
    assert r_raw[off_diagonal].mean() > 0.5

    # Normalized R: no structure, correctly. It sits on the closure baseline
    # rather than exactly 0 -- see closure_null.
    assert r_norm[off_diagonal].mean() == pytest.approx(closure_null(N_OPS), abs=0.05)


def test_shared_factor_loadings_expose_the_confound():
    """The headline diagnostic: if every op loads ~1.0 on the shared per-image
    factor, R is reading which images are hard, and nothing built on it is
    trustworthy."""
    d_grad = _confounded_grid()

    loadings = shared_factor_loadings(d_grad, shared_image_factor(d_grad))

    assert loadings.shape == (N_OPS,)
    assert (loadings > 0.9).all()


def test_normalization_preserves_a_real_correlation():
    """The normalization must remove the shared factor WITHOUT flattening genuine
    redundancy -- otherwise it would just be destroying the signal."""
    rng = np.random.default_rng(1)
    n_images = 800
    base = rng.normal(loc=1.0, scale=0.15, size=(N_OPS, n_images))
    base[5] = base[2] + 0.05 * rng.normal(size=n_images)  # ops 2 and 5 really are alike
    d_grad = base * rng.uniform(0.5, 5.0, size=n_images)

    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    assert r_norm[2, 5] > 0.8
    others = [i for i in range(N_OPS) if i not in (2, 5)]
    assert abs(r_norm[np.ix_(others, others)][np.triu_indices(len(others), 1)].mean()) < 0.2


def test_closure_null_is_the_baseline_not_zero():
    assert closure_null(14) == pytest.approx(-1.0 / 13)
    assert closure_null(1) == 0.0


# --- bootstrap + FDR ---------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_a_naive_column_bootstrap():
    """Columns inside one probe share a batch and a model state, so they are not
    independent draws. Resampling COLUMNS pretends there are N_images independent
    observations and returns intervals that are too tight; resampling whole PROBES
    tells the truth. Same function, different cluster ids."""
    rng = np.random.default_rng(0)
    n_probes, batch = 20, 8
    probe_ids = np.repeat(np.arange(n_probes), batch)

    # A probe-level effect shared by every column of that probe -- precisely the
    # clustering a column bootstrap ignores.
    probe_effect = rng.normal(size=(2, n_probes)).repeat(batch, axis=1)
    d_grad = probe_effect + rng.normal(scale=0.3, size=(2, n_probes * batch))

    lo_probe, hi_probe, _ = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=400)
    lo_col, hi_col, _ = cluster_bootstrap_ci(
        d_grad, np.arange(probe_ids.size), n_reps=400
    )

    assert (hi_probe[0, 1] - lo_probe[0, 1]) > (hi_col[0, 1] - lo_col[0, 1])


def test_bootstrap_ci_narrows_as_more_probes_are_collected():
    def ci_width(n_probes):
        rng = np.random.default_rng(0)
        n_images = n_probes * 4
        d_grad = rng.normal(size=(2, n_images))
        d_grad[1] = 0.6 * d_grad[0] + 0.8 * rng.normal(size=n_images)
        lo, hi, _ = cluster_bootstrap_ci(
            d_grad, np.repeat(np.arange(n_probes), 4), n_reps=300
        )
        return hi[0, 1] - lo[0, 1]

    assert ci_width(120) < ci_width(15)


def test_only_a_genuinely_correlated_pair_survives_fdr():
    """91 pairwise cells means several will look significant by chance. BH is what
    stops a pruning decision resting on one of them."""
    rng = np.random.default_rng(0)
    n_probes, batch = 40, 8
    n_images = n_probes * batch
    probe_ids = np.repeat(np.arange(n_probes), batch)

    d_grad = rng.normal(size=(N_OPS, n_images))
    d_grad[9] = 0.9 * d_grad[3] + np.sqrt(1 - 0.81) * rng.normal(size=n_images)

    _, _, p = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=500)
    q, survives = fdr_correct(p, alpha=0.05)

    assert survives[3, 9], "the injected redundant pair should survive FDR"
    assert q[3, 9] < 0.05
    upper = np.triu_indices(N_OPS, k=1)
    assert survives[upper].sum() <= 3, "too many null pairs slipped through BH"


def test_fdr_is_symmetric_and_ignores_the_diagonal():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(6, 6))
    p = (p + p.T) / 2

    q, survives = fdr_correct(p)

    np.testing.assert_allclose(q, q.T, equal_nan=True)
    assert np.isnan(np.diag(q)).all()
    assert not np.diag(survives).any()


# --- DDP ---------------------------------------------------------------------


def test_merge_rank_buffers_preserves_probe_boundaries():
    """Each rank probes its own shard, so rank 0's R would otherwise see only
    1/world_size of the images. Merged probes stay SEPARATE entries -- they are
    different images, and the bootstrap resamples on those boundaries."""
    rank0 = {name: [np.zeros(4), np.zeros(4)] for name in NAMES}
    rank1 = {name: [np.ones(4)] for name in NAMES}

    merged = merge_rank_buffers([rank0, rank1], NAMES)
    d_grad, probe_ids = stack_probe_buffer(merged, NAMES)

    assert len(merged[NAMES[0]]) == 3
    assert d_grad.shape == (N_OPS, 12)
    assert np.unique(probe_ids).size == 3


# --- serialization -----------------------------------------------------------


def test_jsonable_writes_nan_as_null():
    """R deliberately contains NaN (dropped ops). json.dumps would emit a bare
    `NaN` token, which python reads back but jq and pandas will not."""
    payload = _jsonable({"R": np.array([[1.0, np.nan], [np.nan, 1.0]]), "n": np.int64(4)})

    text = json.dumps(payload)
    assert "NaN" not in text
    assert json.loads(text)["R"] == [[1.0, None], [None, 1.0]]
    assert json.loads(text)["n"] == 4


# --- interval gating ----------------------------------------------------------


class _FakeRunner:
    def __init__(self, work_dir, buffer, max_iters=1000):
        self.cfg = types.SimpleNamespace(work_dir=str(work_dir))
        self.aug_grad_buffer = buffer
        self.max_iters = max_iters
        self.iter = 0


def _hook(**kwargs):
    # interval=500 against max_iters=1000 fires at iter 499 and 999 -- runner.iter is
    # the 0-based index of the iteration that just finished, so those are the 500th
    # and 1000th, i.e. 50% and 100% through.
    kwargs.setdefault("interval", 500)
    kwargs.setdefault("bootstrap", False)
    kwargs.setdefault("n_min", 8)
    return PerturbationSensitivityAnalysisHookWithGradients(**kwargs)


def _fill(buffer, n_probes, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_probes):
        for name in NAMES:
            buffer[name].append(rng.normal(size=batch))


def test_r_is_emitted_only_when_the_interval_fires(tmp_path):
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()

    # Mid-interval: window still open, nothing emitted, nothing thrown away.
    _fill(buffer, n_probes=3)
    runner.iter = 400
    hook.after_train_iter(runner, batch_idx=400)
    assert not (tmp_path / "corr_matrix_log.json").exists()
    assert len(buffer[NAMES[0]]) == 3, "buffer must keep accumulating mid-window"

    # The 500th iteration: emit, then clear so the next window is one model state.
    _fill(buffer, n_probes=3, seed=1)
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    records = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["checkpoint"] == 0.5
    assert record["n_images"] == 24  # 6 probes x 4 images
    assert record["n_probes"] == 6
    assert len(buffer[NAMES[0]]) == 0, "window must be cleared at an emission"


def test_r_is_not_emitted_by_a_val_epoch(tmp_path):
    """The property this pipeline exists for.

    R used to be emitted from ``after_val_epoch``, which under --aug-type=ours is
    called by RobustValLoop itself -- so the correlation matrix was computed inside
    the SA pipeline's val epoch and could only appear on an SA round. The two
    measure different things and now keep separate clocks: a val epoch, even one
    landing exactly on a firing iteration, must produce nothing.
    """
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()

    _fill(buffer, n_probes=6)
    runner.iter = 499  # a firing iteration for after_train_iter

    hook.after_val_epoch(runner)
    assert not (tmp_path / "corr_matrix_log.json").exists(), (
        "the SA pipeline's val epoch must not drive R any more"
    )
    assert len(buffer[NAMES[0]]) == 6, "and it must not consume the window either"

    # The pipeline's own clock still works, from the same state.
    hook.after_train_iter(runner, batch_idx=499)
    assert (tmp_path / "corr_matrix_log.json").exists()


def test_both_halves_of_the_pipeline_share_one_clock(tmp_path):
    """The collector sweeps and the analyser correlates THAT sweep, at the same hook
    point on the same iteration. If the two gates ever disagreed by even one
    iteration, the analyser would drain an empty buffer (or, worse, a stale one) and
    nothing in the output would look wrong. They are literally the same predicate --
    this is what pins that down.
    """
    from sensaug.hooks.grad_hook import CollectGradientHook, fires_at

    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    collector = CollectGradientHook(interval=300)
    analyser = _hook(interval=300)

    fired = []
    for iteration in range(1000):
        runner.iter = iteration
        fired.append(
            (
                fires_at(runner, collector.interval),
                fires_at(runner, analyser.interval),
            )
        )

    assert all(sweeps == emits for sweeps, emits in fired), (
        "the sweep and the emission must fire on exactly the same iterations"
    )
    # 300, 600, 900 -- plus the last iteration, which always fires so that the
    # converged model's R exists even though 1000 % 300 != 0.
    assert [i for i, (sweeps, _) in enumerate(fired) if sweeps] == [299, 599, 899, 999]


def test_a_resumed_run_does_not_re_emit_passed_intervals(tmp_path):
    """The old gate was a `_next_checkpoint` index into a tuple of progress
    fractions, rebuilt at 0 on resume -- so a run resumed at 60% re-emitted the 25%
    and 50% R from a model state those checkpoints never saw. A modulo gate keeps no
    state across the restart."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()

    _fill(buffer, n_probes=6)
    runner.iter = 600  # a fresh hook on a run resumed past the first interval
    hook.after_train_iter(runner, batch_idx=600)
    assert not (tmp_path / "corr_matrix_log.json").exists()

    runner.iter = 999  # the next real firing point
    hook.after_train_iter(runner, batch_idx=999)
    (record,) = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert record["checkpoint"] == 1.0


def test_window_below_n_min_is_skipped_and_discarded(tmp_path):
    """Two data points always lie on a line, so the old '>= 2 probes' gate made
    corrcoef return exactly +/-1 for every pair and logged it as real. A window
    that is too small is dropped, NOT rolled into the next checkpoint -- that
    would pool two different model states."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(n_min=100)

    _fill(buffer, n_probes=2)  # 8 images, well under n_min
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    assert not (tmp_path / "corr_matrix_log.json").exists()
    assert len(buffer[NAMES[0]]) == 0, "an undersized window is discarded, not carried"


def test_record_carries_both_matrices_and_the_confound_diagnostic(tmp_path):
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(normalize_per_image=True)

    _fill(buffer, n_probes=6)
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    (record,) = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert record["names"] == NAMES
    assert np.array(record["R_raw"], dtype=float).shape == (N_OPS, N_OPS)
    assert np.array(record["R_scalenorm"], dtype=float).shape == (N_OPS, N_OPS)
    assert len(record["shared_factor_loadings"]) == N_OPS
    assert record["dropped"] == []


def test_bootstrap_detail_goes_to_its_own_file(tmp_path):
    """91 cells x several fields per checkpoint would bury R if inlined."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(bootstrap=True, bootstrap_reps=50)

    _fill(buffer, n_probes=8)
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    record = json.loads((tmp_path / "corr_bootstrap_log.txt").read_text().strip())
    assert record["cluster_level"] == "probe"
    assert len(record["cells"]) == N_OPS * (N_OPS - 1) // 2  # 91
    cell = record["cells"][0]
    assert {"i", "j", "r", "ci_lo", "ci_hi", "ci_width", "p", "q", "survives_fdr"} <= set(cell)
    # R itself stays in the other file, ungarbled by all of this
    (r_record,) = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert "R_raw" in r_record


# --- the json envelope -------------------------------------------------------


def test_corr_json_accumulates_emissions_as_one_array(tmp_path):
    """Each emission rewrites the whole file, so the earlier ones have to survive the
    rewrite -- the append-only JSONL this replaces got that for free."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()  # interval=500 -> fires at iter 499 and 999

    _fill(buffer, n_probes=6)
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    _fill(buffer, n_probes=6, seed=1)
    runner.iter = 999
    hook.after_train_iter(runner, batch_idx=999)

    records = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert [record["checkpoint"] for record in records] == [0.5, 1.0]


def test_a_dropped_op_survives_the_json_round_trip_as_null(tmp_path):
    """A dropped op's row and column of R are NaN, and json.dump writes a bare
    `NaN` token for those -- not valid JSON. _jsonable maps them to null, and the
    writer passes allow_nan=False so that a value which slipped past _jsonable
    fails the write instead of quietly emitting a file that only looks like JSON.
    Nothing else in this module exercises a NaN, so this is the test holding that
    contract down."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()

    _fill(buffer, n_probes=6)
    # Flatten one op's sensitivity to a constant: zero variance, so correlate()
    # drops it rather than dividing by a zero std.
    flat = NAMES[0]
    buffer[flat] = [np.zeros_like(probe) for probe in buffer[flat]]

    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)

    (record,) = json.loads((tmp_path / "corr_matrix_log.json").read_text())
    assert record["dropped"] == [flat]
    assert record["R_raw"][0] == [None] * N_OPS, "the dropped row must be null, not NaN"
    assert all(row[0] is None for row in record["R_raw"])
    assert record["R_raw"][1][1] == 1.0, "a kept op still has a real diagonal"


# --- the redundancy handoff ---------------------------------------------------
#
# prune_augmentations does not prune. It publishes a per-op redundancy score onto
# the runner for RobustValLoop to reweight the training pdf by. The scoring itself
# is tested in test_redundancy.py against synthetic input; these cover the handoff
# -- what gets published, and the three cases where nothing must be.


def _redundancy_hook(tmp_path, **kwargs):
    kwargs.setdefault("interval", 500)
    kwargs.setdefault("bootstrap", False)
    kwargs.setdefault("n_min", 8)
    return _hook(**kwargs)


def _emit_once(tmp_path, hook, seed=0, n_probes=6):
    """Drive one full emission through the interval gate, as training would."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    _fill(buffer, n_probes=n_probes, seed=seed)
    runner.iter = 499
    hook.after_train_iter(runner, batch_idx=499)
    return runner


def test_an_emission_publishes_a_redundancy_score(tmp_path):
    """The handoff channel: the hook fires on corr_interval and the SA loop reads
    on round_interval, so the score is left on the runner for whenever the loop
    next looks rather than pushed at it."""
    runner = _emit_once(tmp_path, _redundancy_hook(tmp_path))

    published = runner.corr_redundancy
    assert published is not None
    assert set(published["red"]) == set(NAMES)
    assert published["mode"] == "squared"
    assert published["checkpoint"] == 0.5
    assert published["fdr_gated"] is False, "no bootstrap was run"
    assert all(np.isfinite(v) for v in published["red"].values())


def test_the_published_score_is_standardized(tmp_path):
    """What makes one lambda portable across runs and checkpoints. If the raw row
    sums were published instead, a lambda tuned on one checkpoint would mean
    something different at the next."""
    runner = _emit_once(tmp_path, _redundancy_hook(tmp_path))
    values = np.array(list(runner.corr_redundancy["red"].values()))

    assert values.mean() == pytest.approx(0.0, abs=1e-9)
    assert values.std() == pytest.approx(1.0, rel=1e-6)


def test_the_score_is_appended_to_its_own_jsonl_log(tmp_path):
    """One record per emission, appended -- so a --resume adds to the history
    rather than rewriting it, matching corr_bootstrap_log.txt."""
    hook = _redundancy_hook(tmp_path)
    _emit_once(tmp_path, hook)
    _emit_once(tmp_path, hook, seed=1)

    lines = (tmp_path / "corr_redundancy_log.txt").read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert set(record["red"]) == set(NAMES)
    assert "raw" in record and "dropped" in record


def test_the_raw_row_sums_are_logged_alongside_the_standardized_score(tmp_path):
    """Standardization is lossy -- it throws away the scale that says how redundant
    the bank is overall. Recording the raw sums keeps that recoverable offline."""
    runner = _emit_once(tmp_path, _redundancy_hook(tmp_path))
    raw = np.array(list(runner.corr_redundancy["raw"].values()))

    assert (raw >= 0).all(), "squared row sums cannot be negative"
    assert raw.std() > 0


def test_no_score_is_published_when_the_shared_factor_alarm_fires(tmp_path):
    """_emit already warns 'Do not act on it' when every op loads on one per-image
    factor -- at that point R ranks which IMAGES are hard, not which augmentations
    overlap. Until now nothing enforced it and prune_augmentations ran anyway."""
    hook = _redundancy_hook(tmp_path)
    runner = _FakeRunner(tmp_path, {name: [] for name in NAMES}, max_iters=1000)
    runner.corr_redundancy = {"stale": True}

    r = np.eye(N_OPS)
    hook.prune_augmentations(runner, r, checkpoint=0.5, max_loading=0.95)

    assert runner.corr_redundancy is None, "a score was published past the alarm"
    assert not (tmp_path / "corr_redundancy_log.txt").exists()


def test_no_score_is_published_when_red_is_degenerate(tmp_path):
    """A silent no-op is the failure mode most likely to waste a week: the run
    looks like a real experimental arm and the pdf never moved."""
    hook = _redundancy_hook(tmp_path)
    runner = _FakeRunner(tmp_path, {name: [] for name in NAMES}, max_iters=1000)

    # Identity R: every off-diagonal cell zero, so every op is equally redundant.
    hook.prune_augmentations(runner, np.eye(N_OPS), checkpoint=0.5, max_loading=0.1)

    assert runner.corr_redundancy is None


def test_an_all_nan_matrix_publishes_nothing_and_does_not_raise(tmp_path):
    hook = _redundancy_hook(tmp_path)
    runner = _FakeRunner(tmp_path, {name: [] for name in NAMES}, max_iters=1000)

    hook.prune_augmentations(runner, np.full((N_OPS, N_OPS), np.nan), checkpoint=0.5)

    assert getattr(runner, "corr_redundancy", None) is None


def test_the_fdr_survivor_mask_is_used_when_the_bootstrap_ran(tmp_path):
    """The gate section 3.2 asks for: cells that did not survive multiplicity
    correction at 496 simultaneous tests contribute nothing to red(a)."""
    hook = _redundancy_hook(tmp_path, bootstrap=True, bootstrap_reps=50)
    runner = _emit_once(tmp_path, hook, n_probes=40)

    assert runner.corr_redundancy["fdr_gated"] is True


def test_red_mode_selects_the_reduction(tmp_path):
    """squared/abs/signed are the ablation arms. They have to actually differ."""
    # Separate work_dirs: each hook caches its log paths on first emission.
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir(exist_ok=True)

    squared = _emit_once(tmp_path / "a", _redundancy_hook(tmp_path)).corr_redundancy
    hook = _redundancy_hook(tmp_path, red_mode="signed")
    signed = _emit_once(tmp_path / "b", hook).corr_redundancy

    assert signed["mode"] == "signed"
    assert squared["red"] != signed["red"]


def test_an_unknown_red_mode_is_rejected_at_construction(tmp_path):
    """Not at the first emission, which on the default corr_interval is 25% of the
    way into a multi-hour run."""
    with pytest.raises(ValueError, match="unknown red_mode"):
        PerturbationSensitivityAnalysisHookWithGradients(interval=10, red_mode="cubed")
