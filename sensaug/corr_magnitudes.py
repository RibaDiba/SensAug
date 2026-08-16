"""Per-augmentation magnitude distributions, handed from the sensitivity-analysis
pipeline to the gradient cross-correlation probe.

WHY THIS EXISTS
---------------
The probe used to differentiate every op at one hardcoded ``ref_magnitude=0.5``.
That constant is not commensurable across ops: for the 12 photometric ops it means
"halfway to the rail", but for ``blur`` the magnitude IS sigma (0.5 is a mild
blur) and for ``noise`` it is a std that saturates most pixels. Measured on a
random image, magnitude 0.5 moves a pixel by at most 78/255 under ``blur`` and
251/255 under ``noise`` -- so every cell of R was a correlation between two ops
held at arbitrary, mutually incomparable strengths.

The SA pipeline already estimates, per op, which magnitudes matter for the
current model. This module turns its output into the magnitudes the probe should
use, so R describes the augmentations as they are actually applied.

THE HANDOFF
-----------
``RobustValLoop`` publishes a snapshot onto the runner after each SA round;
``CollectGradientHook`` reads whatever is there at fire time. That is why the
correlation pipeline "uses the latest distribution" when it fires less often than
the SA loop -- there is no queue, only the current value.

Deliberately free of any mmcv/mmseg/torch dependency: this is the part of the
handoff with real logic in it, so it stays unit-testable without the OpenMMLab
stack installed.
"""

import numpy as np

# Magnitude selection modes. See CollectGradientHook for what each one does to R.
MODE_MODAL = "mode"
MODE_SAMPLED_SHARED = "sampled_shared"
MODE_SAMPLED_INDEPENDENT = "sampled_independent"
MODE_FIXED = "fixed"
MAGNITUDE_MODES = (
    MODE_MODAL,
    MODE_SAMPLED_SHARED,
    MODE_SAMPLED_INDEPENDENT,
    MODE_FIXED,
)

# The Gaussian jitter RandomTrainTransformNew adds to a sampled level before
# applying it (augmentations.py). Mirrored here so the sampled modes reproduce
# the magnitudes training actually uses, not just the pdf's support points.
LEVEL_JITTER_STD = 0.1

# Magnitudes are clipped to this, matching both the training transform's clip and
# the differentiable ops' [0, 1] magnitude contract.
LEVEL_MIN, LEVEL_MAX = 0.0, 1.0


def conditional_levels(pdf_dict, op_names=None):
    """Joint pdf over ``(op, level)`` pairs -> per-op conditional over levels.

    ``pdf_dict`` (from RobustValLoop.generate_pdf_*) is a JOINT distribution: its
    values sum to 1 across all ops together, so an op's raw values are not a
    distribution over that op's own levels. The probe needs "given this op, which
    magnitude", so each op's block is renormalized to sum to 1 on its own.

    The synthetic ``("none", 0)`` entry is dropped -- it is training's
    probability of applying no augmentation at all, not a magnitude for any op.

    Returns ``{op: {"levels": [...], "probs": [...]}}``, levels ascending.
    """
    by_op = {}
    for key, prob in pdf_dict.items():
        op, level = key
        if op == "none":
            continue
        if op_names is not None and op not in op_names:
            continue
        by_op.setdefault(op, []).append((float(level), float(prob)))

    snapshot = {}
    for op, pairs in by_op.items():
        pairs.sort()  # ascending level, so the arrays are reproducible
        levels = np.asarray([level for level, _ in pairs], dtype=np.float64)
        probs = np.asarray([prob for _, prob in pairs], dtype=np.float64)

        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            # An op whose every level got zero mass carries no preference. Uniform
            # is the honest reading; renormalizing by zero would be NaN and would
            # propagate silently into every magnitude drawn for that op.
            probs = np.full(probs.shape, 1.0 / probs.size)

        snapshot[op] = {"levels": levels.tolist(), "probs": probs.tolist()}
    return snapshot


def modal_magnitude(entry):
    """The single most probable level for one op -- the default probe magnitude.

    Constant across the batch, which is what keeps R a statement about image
    variation alone: a magnitude that varied per image would add a second source
    of variance to every row.

    NOTE which level this is depends on the pdf's shape. generate_pdf_new weights
    levels by betabinom(0.75, 1.0) over their mIoU rank, which is DECREASING, so
    the mode is the rank-0 level -- under the default (descending_MA=False,
    reverse=True) that is the HIGHEST-mIoU, i.e. mildest, level. Check the first
    emitted snapshot against expectation before trusting a run.
    """
    return float(entry["levels"][int(np.argmax(np.asarray(entry["probs"])))])


def sample_magnitudes(
    entry,
    n,
    rng=None,
    quantiles=None,
    jitter_draw=None,
    jitter_std=LEVEL_JITTER_STD,
):
    """Draw `n` magnitudes for one op, the way training would.

    Mirrors RandomTrainTransformNew: pick a level from the op's distribution, add
    N(0, jitter_std), clip to [0, 1].

    `quantiles` and `jitter_draw` are the common-random-numbers hooks. Pass the
    SAME arrays for every op (mode ``sampled_shared``) and image i gets the same
    percentile of each op's level distribution and the same jitter deviate
    everywhere. Leave them None (mode ``sampled_independent``) and every op draws
    on its own, which is what training literally does per image -- but independent
    draws inject noise that is uncorrelated across rows, attenuating every cell of
    R toward zero by a factor that depends on the pdf's spread, so R stops being
    comparable across checkpoints.
    """
    levels = np.asarray(entry["levels"], dtype=np.float64)
    probs = np.asarray(entry["probs"], dtype=np.float64)

    if quantiles is None:
        if rng is None:
            raise ValueError("sample_magnitudes needs an rng when quantiles is None")
        index = rng.choice(levels.size, size=n, p=probs)
    else:
        # Inverse-CDF: searchsorted on the cumulative probabilities maps a uniform
        # quantile onto a level index. Shared quantiles therefore mean "the same
        # percentile of each op's distribution", not "the same index" -- ops with
        # differently shaped pdfs still get their own magnitudes.
        index = np.searchsorted(np.cumsum(probs), np.asarray(quantiles), side="left")
        index = np.clip(index, 0, levels.size - 1)

    drawn = levels[index]

    if jitter_std:
        if jitter_draw is None:
            if rng is None:
                raise ValueError("sample_magnitudes needs an rng to jitter")
            jitter_draw = rng.normal(0.0, 1.0, size=n)
        drawn = drawn + np.asarray(jitter_draw) * jitter_std

    return np.clip(drawn, LEVEL_MIN, LEVEL_MAX)


def resolve_magnitudes(snapshot, op_names, batch_size, mode, fallback, rng=None):
    """Magnitudes for every op for one probe batch.

    Returns ``{op: np.ndarray of shape (batch_size,)}``. Ops absent from the
    snapshot fall back to `fallback` -- an op the SA curve does not cover must not
    silently vanish from R.

    Returns the fallback for EVERY op when `snapshot` is empty or `mode` is
    ``fixed``, which is the pre-first-SA and SA-disabled case.
    """
    if mode not in MAGNITUDE_MODES:
        raise ValueError(f"unknown magnitude mode {mode!r}, expected one of {MAGNITUDE_MODES}")

    constant = np.full(batch_size, float(fallback), dtype=np.float64)
    if mode == MODE_FIXED or not snapshot:
        return {op: constant.copy() for op in op_names}

    if mode == MODE_MODAL:
        return {
            op: (
                np.full(batch_size, modal_magnitude(snapshot[op]), dtype=np.float64)
                if op in snapshot
                else constant.copy()
            )
            for op in op_names
        }

    # Both sampled modes. For sampled_shared, draw the quantile and the jitter ONCE
    # here and reuse them for every op -- that shared draw is the whole difference
    # between the two modes.
    shared = mode == MODE_SAMPLED_SHARED
    quantiles = rng.random(batch_size) if shared else None
    jitter_draw = rng.normal(0.0, 1.0, size=batch_size) if shared else None

    return {
        op: (
            sample_magnitudes(
                snapshot[op],
                batch_size,
                rng=rng,
                quantiles=quantiles,
                jitter_draw=jitter_draw,
            )
            if op in snapshot
            else constant.copy()
        )
        for op in op_names
    }
