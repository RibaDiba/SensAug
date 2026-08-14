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
    """
    Convert a joint operation-and-level distribution into per-operation conditional distributions.
    
    Parameters:
        pdf_dict (dict): Joint probabilities keyed by operation and level.
        op_names (iterable, optional): Operations to include in the result.
    
    Returns:
        dict: Mapping of each operation to ascending ``levels`` and normalized ``probs``.
            The synthetic ``"none"`` entry is excluded. Operations with zero total
            probability receive a uniform level distribution.
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
    """
    Selects the most probable magnitude level for an operation.
    
    Parameters:
    	entry (dict): Operation distribution containing `levels` and `probs`.
    
    Returns:
    	float: The level with the highest probability.
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
    """
    Draw magnitudes for one operation from its level distribution.
    
    Parameters:
        entry (dict): Operation entry containing ``levels`` and ``probs``.
        n (int): Number of magnitudes to draw.
        rng: Random number generator used when quantiles or jitter values are not supplied.
        quantiles: Optional uniform quantiles used for inverse-CDF level selection.
        jitter_draw: Optional standard-normal deviates used for magnitude jitter.
        jitter_std (float): Standard deviation of the Gaussian jitter.
    
    Returns:
        numpy.ndarray: Magnitudes clipped to the range [0, 1].
    
    Raises:
        ValueError: If random sampling or jitter is required but no random number
            generator is supplied.
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
    """
    Resolve per-operation magnitudes for a probe batch.
    
    Parameters:
        snapshot: Per-operation magnitude distributions.
        op_names: Operations for which to produce magnitudes.
        batch_size: Number of magnitudes to generate for each operation.
        mode: Magnitude selection mode.
        fallback: Magnitude used for fixed mode, empty snapshots, and operations absent from the snapshot.
    
    Returns:
        A mapping from operation names to arrays of shape ``(batch_size,)``.
    
    Raises:
        ValueError: If ``mode`` is unsupported or sampling requires an RNG that was not supplied.
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
