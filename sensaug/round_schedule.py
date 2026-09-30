"""The SA val-round grid, and where on it the correlation pipeline fires.

Pure integers and yaml -- no torch, no mmseg, no mmengine -- so the schedule can be
unit-tested without a GPU and read by an offline script. Same posture as
sensaug/redundancy.py, and for the same reason: this is the part of the pipeline
that is easiest to get subtly wrong and cheapest to test.

WHAT A ROUND IS. `RobustValLoop` runs once every `round_interval` training
iterations and calls that a round (`sensaug/loops/sensaug_loop.py`). It spends the
first `warmup_rounds` of them training on the initial uniform policy, then starts
generating a pdf; the SA *curve* that pdf is derived from is recomputed every
SA_CURVE_CADENCE rounds thereafter.

WHY THE CORRELATION PIPELINE USES THAT GRID. R exists to be read by the pdf, and
the pdf only changes shape when the SA curve behind it does. Emitting on a clock of
its own -- which is what `corr_interval` was, `max_iters // 4` -- put matrices
wherever a quarter of training happened to land: measured mid-curve, unread for
several rounds, and in the case of the final emission (at `max_iters`) never read
at all, because training ends before another pdf is built.

So the default schedule is `derive_corr_rounds`: one baseline probe in the last
warmup round, then one emission per SA-curve recompute. `configs/rounds.yaml` can
override any of it. `--corr-interval` still buys back the independent clock.
"""

import os
from dataclasses import dataclass

import yaml

__all__ = [
    "SA_CURVE_CADENCE",
    "DEFAULT_ROUNDS_CONFIG",
    "RoundConfig",
    "Schedule",
    "load_round_config",
    "derive_corr_rounds",
    "sa_curve_recompute_rounds",
    "control_rounds",
    "round_fire_iters",
    "resolve_schedule",
]

#: Rounds between SA-curve recomputes -- the `% SA_CURVE_CADENCE` in
#: `RobustValLoop.run`. Deliberately NOT a config key: a file that could move this
#: without moving the val loop would push the sweeps off the very recompute rounds
#: they exist to align with. Defined here and imported there, so the number lives
#: in one place rather than two that have to agree.
SA_CURVE_CADENCE = 6

#: Repo-relative default for `--rounds-config`.
DEFAULT_ROUNDS_CONFIG = "configs/rounds.yaml"

#: Keys `configs/rounds.yaml` may contain. Anything else raises: a typo'd key that
#: silently does nothing is exactly how a schedule change gets written up as
#: "had no effect".
_KEYS = {"n_rounds", "warmup_rounds", "corr_rounds", "control_rounds", "corr_cadence"}


@dataclass(frozen=True)
class RoundConfig:
    """`configs/rounds.yaml`, parsed but not yet resolved against a run.

    `corr_rounds` / `control_rounds` stay None when the file leaves them null,
    which is what `resolve_schedule` reads as "derive them". `corr_cadence` stays
    None the same way, which `resolve_schedule` reads as "use SA_CURVE_CADENCE".
    """

    n_rounds: int
    warmup_rounds: int
    corr_rounds: list = None
    control_rounds: list = None
    corr_cadence: int = None
    path: str = "<defaults>"


@dataclass(frozen=True)
class Schedule:
    """One run's resolved correlation schedule.

    `rounds` is for humans and the launch log; `fire_iters` / `control_iters` /
    `full_recheck_iters` are what the hooks gate on.
    """

    rounds: tuple
    fire_iters: tuple
    control_iters: tuple
    derived: bool
    #: The subset of `fire_iters` that land on an SA-curve recompute round --
    #: where `--corr-skip-pruned-eval` forces CollectGradientHook to measure
    #: every op regardless of runner.corr_pruned_ops, rather than reading a
    #: live "was the curve just recomputed" signal that would lag by one round
    #: relative to that iteration's own val loop (see resolve_schedule).
    #: Always a subset of `fire_iters`, by construction (an intersection).
    #: Empty when `fire_iters` never lands on a recompute round -- e.g. a
    #: hand-pinned `corr_rounds` in configs/rounds.yaml that avoids them.
    full_recheck_iters: tuple = ()
    warnings: tuple = ()


def load_round_config(path: str) -> RoundConfig:
    """Parse a rounds YAML.

    Fails loudly on a missing or malformed file rather than degrading to built-in
    defaults: this file decides where every gradient sweep in the run lands, and a
    silent fallback would produce a perfectly plausible experiment measuring
    something other than what the launch command says. Same stance as
    `CollectGradientHook._load_seed_snapshot` takes on `--corr-magnitudes`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"rounds config {path!r} does not exist. The repo ships "
            f"{DEFAULT_ROUNDS_CONFIG}; pass --rounds-config to point elsewhere."
        )
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")

    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}. Known keys are {sorted(_KEYS)}."
        )
    for key in ("n_rounds", "warmup_rounds"):
        if raw.get(key) is None:
            raise ValueError(f"{path}: {key} is required and must not be null")

    return RoundConfig(
        n_rounds=int(raw["n_rounds"]),
        warmup_rounds=int(raw["warmup_rounds"]),
        corr_rounds=_as_round_list(raw.get("corr_rounds"), "corr_rounds", path),
        control_rounds=_as_round_list(raw.get("control_rounds"), "control_rounds", path),
        corr_cadence=_as_cadence(raw.get("corr_cadence"), path),
        path=path,
    )


def _as_round_list(value, key: str, path: str):
    """A null-or-list-of-ints config value, normalized and sorted.

    An empty list is NOT the same as null: it means "fire nowhere", which
    `resolve_schedule` rejects, while null means "derive it".
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}: {key} must be a list of round numbers or null")
    rounds = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{path}: {key} contains a non-integer round {item!r}")
        rounds.append(int(item))
    return sorted(set(rounds))


def _as_cadence(value, path: str):
    """A null-or-positive-int cadence override, or None to keep SA_CURVE_CADENCE.

    Unlike `corr_rounds`/`control_rounds`, this decouples the correlation
    pipeline's OWN emission spacing from the SA curve's recompute cadence
    (`SA_CURVE_CADENCE`, still fixed) -- see `derive_corr_rounds`.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{path}: corr_cadence must be a positive integer round count or "
            f"null, got {value!r}"
        )
    return value


def derive_corr_rounds(
    n_rounds: int, warmup_rounds: int, cadence: int = SA_CURVE_CADENCE
) -> list:
    """The default schedule: which SA rounds the correlation pipeline fires on.

    - ``warmup_rounds - 1`` -- the CONTROL probe. The last warmup round, so R is
      measured on a model the pdf has never touched: the baseline every later
      matrix is read against. It does not feed the pdf, and could not -- at that
      round the SA loop is still in warmup and generates no pdf at all.
      `control_rounds` is what marks it, and the analyser withholds the publication
      rather than relying on that ordering.
    - ``warmup_rounds + k*cadence`` -- the SA-curve recompute rounds. Firing HERE
      rather than on an interval that lands wherever is what makes an emission
      actionable: the sweep runs from ``after_train_iter``, which
      IterBasedTrainLoop reaches BEFORE it calls ``val_loop.run()``, so the R
      measured at round r is already on the runner when round r builds its pdf,
      and it stays current for exactly the ``cadence`` rounds that curve governs.

    The final round is excluded: training ends with it, so nothing would ever read
    the R it produced.

    At the defaults (20 rounds, 4 warmup, cadence 6) this is rounds 3, 4, 10, 16 --
    the round-3 baseline, then one emission per SA curve, governing rounds 4-9,
    10-15 and 16-18.

    With ``--no-warmup`` (warmup_rounds=0) there is no control probe: training is
    pdf-driven from round 0, so no round of that run is a clean baseline.

    A ``cadence`` other than ``SA_CURVE_CADENCE`` (via ``configs/rounds.yaml``'s
    ``corr_cadence``, see ``resolve_schedule``) does NOT move the SA curve's own
    recompute point -- that is still every ``SA_CURVE_CADENCE`` rounds, unchanged.
    A denser cadence just adds emissions that land *inside* an already-current
    curve window: the pdf's shape doesn't change on those rounds, but `red(a)` is
    refreshed from a fresh gradient sweep, so the redundancy score can move even
    though the SA curve behind it hasn't.
    """
    if cadence < 1:
        raise ValueError(f"cadence must be a positive round count, got {cadence}")
    rounds = []
    if warmup_rounds > 0:
        rounds.append(warmup_rounds - 1)
    rounds.extend(range(warmup_rounds, max(n_rounds - 1, 0), cadence))
    return [r for r in rounds if 0 <= r < n_rounds - 1]


def sa_curve_recompute_rounds(
    n_rounds: int, warmup_rounds: int, cadence: int = SA_CURVE_CADENCE
) -> list:
    """Which rounds the SA curve itself recomputes on -- the exact
    ``(round - warmup_rounds) % SA_CURVE_CADENCE == 0`` gate in
    ``RobustValLoop.run()``, restated here as a set of round numbers.

    ALWAYS called with the default `cadence` (`SA_CURVE_CADENCE`) by
    `resolve_schedule` -- never with a run's `corr_cadence`, which controls how
    often the CORRELATION pipeline re-emits, not how often the curve itself
    refreshes. The two are independent on purpose (see `derive_corr_rounds`'s
    docstring), so this function must not be handed the wrong one: a denser
    `corr_cadence` inserts emissions between recompute rounds, it does not move
    them.

    Exists so "which of the correlation pipeline's firing rounds happen to be a
    curve recompute" is schedule arithmetic, fixed before training starts,
    rather than something a hook has to infer from live runner state at
    sweep-time -- see `resolve_schedule`'s `full_recheck_iters`.
    """
    if cadence < 1:
        raise ValueError(f"cadence must be a positive round count, got {cadence}")
    return [
        r
        for r in range(warmup_rounds, n_rounds, cadence)
        if 0 <= r < n_rounds
    ]


def control_rounds(rounds, warmup_rounds: int) -> list:
    """The subset of `rounds` whose R is a baseline and must not reach the pdf.

    A firing round inside the warmup window, i.e. exactly the ``warmup_rounds - 1``
    probe above. Derived from the schedule rather than hardcoded to one round so
    that "measured before the pdf existed" stays the definition.
    """
    return [r for r in rounds if r < warmup_rounds]


def round_fire_iters(rounds, round_interval: int, pre_train_round: bool = False):
    """The iteration counts at which those rounds' val loops are about to run.

    Round r's val loop runs at iteration count ``(r + 1) * round_interval``:
    IterBasedTrainLoop increments ``_iter`` AFTER ``after_train_iter`` and then
    tests ``_iter % val_interval``, so the hook point immediately preceding round
    r's val loop is the one where ``iteration_count()`` equals that. Firing one hook
    point before the loop that consumes it is the whole reason this mapping is
    exact rather than approximate.

    ``pre_train_round`` covers RobustIterBasedTrainLoop's ``init_sa``, which runs
    one val round BEFORE the training loop starts (``--resume``, ``--no-warmup``).
    That round consumes round number 0 at iteration 0, shifting every later round
    one interval earlier. Rounds that then land at or before iteration 0 are
    dropped: they happen before any training iteration, so there is no hook point
    at which they could fire.
    """
    offset = 1 if pre_train_round else 0
    return tuple(
        (r + 1 - offset) * round_interval for r in rounds if r + 1 - offset >= 1
    )


def resolve_schedule(
    cfg: RoundConfig,
    n_rounds: int,
    warmup_rounds: int,
    round_interval: int,
    pre_train_round: bool = False,
    cadence: int = None,
) -> Schedule:
    """Turn a parsed config plus this run's actual round grid into firing iterations.

    NOTE `n_rounds` and `warmup_rounds` are the run's EFFECTIVE values, not
    `cfg.n_rounds` / `cfg.warmup_rounds`. The config's `n_rounds` is the default
    divisor for `round_interval` (`max_iters // n_rounds`), but `--round_interval`
    and `schedule.round_interval` can override that, and `--no-warmup` forces the
    warmup to 0. Building the schedule off the configured numbers instead of the
    ones the run will really use would point every round at the wrong iteration on
    exactly those runs -- silently, since the rounds themselves would still look
    right in the log.

    `cadence` is the opposite: unlike `n_rounds`/`warmup_rounds` there is no CLI
    override for it, so `cfg.corr_cadence` (from `configs/rounds.yaml`) IS the
    effective value when this parameter is left at its default of None. Passing
    `cadence` explicitly (as the tests do) still wins over both, for callers that
    want to bypass the config entirely.

    Returns a `Schedule`; `warnings` carries anything the caller should log but
    which is not worth refusing to train over.
    """
    where = cfg.path
    effective_cadence = (
        cadence
        if cadence is not None
        else (cfg.corr_cadence if cfg.corr_cadence is not None else SA_CURVE_CADENCE)
    )
    if n_rounds < 1:
        raise ValueError(
            f"{where}: a run needs at least one val round, got n_rounds={n_rounds} "
            f"(max_iters // round_interval)"
        )
    if warmup_rounds < 0:
        raise ValueError(f"{where}: warmup_rounds must be >= 0, got {warmup_rounds}")
    if warmup_rounds >= n_rounds:
        raise ValueError(
            f"{where}: warmup_rounds={warmup_rounds} leaves no post-warmup round of "
            f"the {n_rounds} this run has -- the SA loop would never generate a pdf"
        )

    derived = cfg.corr_rounds is None
    rounds = (
        derive_corr_rounds(n_rounds, warmup_rounds, effective_cadence)
        if derived
        else list(cfg.corr_rounds)
    )
    controls = (
        control_rounds(rounds, warmup_rounds)
        if cfg.control_rounds is None
        else list(cfg.control_rounds)
    )

    if not rounds:
        raise ValueError(
            f"{where}: the schedule fires nowhere. This run has only {n_rounds} "
            f"round(s), {warmup_rounds} of them warmup, which is too few for the "
            f"every-{cadence}-round grid. Lower --round_interval to get more rounds, "
            f"or pass --corr-interval to put the correlation pipeline on its own clock."
        )

    _reject_out_of_grid(rounds, n_rounds, "corr_rounds", where)
    _reject_out_of_grid(controls, n_rounds, "control_rounds", where)

    missing = sorted(set(controls) - set(rounds))
    if missing:
        raise ValueError(
            f"{where}: control_rounds {missing} are not in corr_rounds {rounds}. A "
            f"baseline that never fires is a silently missing baseline, not a "
            f"disabled one."
        )

    # Deliberately SA_CURVE_CADENCE, not effective_cadence: which of THIS run's
    # firing rounds happen to be a curve recompute is independent of how dense
    # corr_cadence made the firing schedule -- see sa_curve_recompute_rounds's
    # docstring.
    full_recheck_rounds = sorted(
        set(rounds) & set(sa_curve_recompute_rounds(n_rounds, warmup_rounds))
    )

    warnings = []
    # Allowed, because an explicit list is a deliberate statement, but worth saying
    # out loud: training ends with the final round, so nothing reads the R it emits.
    for r in sorted(set(rounds)):
        if r == n_rounds - 1:
            warnings.append(
                f"{where}: round {r} is the final round of this run, so the R it "
                f"emits is measured after the last pdf is built and no training "
                f"iteration can read it."
            )
    # Whether an empty full_recheck_rounds is worth a warning depends on
    # whether --corr-skip-pruned-eval was even requested, which this module
    # deliberately knows nothing about -- that check lives in train.py.

    return Schedule(
        rounds=tuple(sorted(set(rounds))),
        fire_iters=round_fire_iters(rounds, round_interval, pre_train_round),
        control_iters=round_fire_iters(controls, round_interval, pre_train_round),
        derived=derived,
        full_recheck_iters=round_fire_iters(
            full_recheck_rounds, round_interval, pre_train_round
        ),
        warnings=tuple(warnings),
    )


def _reject_out_of_grid(rounds, n_rounds: int, key: str, where: str) -> None:
    """A round outside [0, n_rounds) has no val loop and so no hook point: it can
    never fire, and would otherwise just be absent from the logs."""
    bad = sorted(r for r in rounds if r < 0 or r >= n_rounds)
    if bad:
        raise ValueError(
            f"{where}: {key} contains round(s) {bad}, but this run has rounds "
            f"0..{n_rounds - 1}. Those would never fire."
        )
