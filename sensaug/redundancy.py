"""Redundancy scoring and max-entropy reweighting of the training pdf.

Turns the gradient cross-correlation matrix R (sensaug/hooks/grad_sens_analysis.py)
into one number per augmentation -- how redundant it is with the rest of the bank --
and folds that into the sampling distribution the SA loop produces.

The rule::

    q(a) proportional to  pdf_old(a) * exp(-lambda * red(a))

is the closed-form solution to: minimise KL(q || pdf_old) subject to
sum_a q(a) red(a) <= C and sum_a q(a) = 1, with lambda the Lagrange multiplier on
the redundancy budget. So this is a derivation rather than a heuristic, and the
lambda=0 arm is exactly the unmodified pdf -- not an approximation of it.

Three things worth knowing before reading the code:

* **red(a) is standardized, and that is load-bearing.** Writing red(a) = c + s(a)
  with c the mean across ops, exp(-lambda*red) = exp(-lambda*c) * exp(-lambda*s),
  and the constant factor cancels in the normaliser. Only deviations from the mean
  ever affect the output. The raw row sums are not comparable across runs -- on the
  logged matrices their means span -0.71 to +4.08 and their standard deviations
  0.24 to 3.69 -- so a lambda tuned against one checkpoint would mean something
  entirely different at the next. Dividing by the standard deviation absorbs that
  into lambda' = lambda/sigma and makes one lambda portable. Measured on the four
  logged corr_matrix_log.json files, lambda=0.25 gives a max/min spread of 2.3-3.1x
  at every checkpoint of both the 14-op and 32-op vocabularies.

* **Down-weighting is soft, never zero.** exp() is strictly positive, so this is
  structural rather than a clamp. At the correlation sizes actually observed here
  (mean |r| of 0.11-0.22) deleting an op would not be justified.

* **The "no augmentation" mass is held fixed.** generate_pdf_new reserves 1/(A+1)
  of the pdf for ("none", 0). Reweighting that entry too would let this mechanism
  change how OFTEN augmentation happens, which is a different intervention from
  changing WHICH augmentation happens, and the two would be inseparable in the
  results. Only the perturbation mass is redistributed.

Pure numpy on purpose: no mmseg, no torch, no registries. It is the piece that has
to be verifiable without a GPU or a training run.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

_STD_FLOOR = 1e-8
# Half-width of the log-space window the exponent is confined to, so the widest
# achievable max/min ratio is exp(2 * _EXP_CLIP) ~ 4e260 -- comfortably inside
# float64's positive normal range, and about 260 orders of magnitude past any
# useful setting (lambda=1 on a standardized score spans about 11).
#
# It exists so that "soft, never zero" survives contact with floating point. exp()
# is strictly positive in exact arithmetic, but a wide enough exponent underflows
# to exactly 0 after normalisation, which would silently turn down-weighting into
# the hard deletion this design rules out. Past the clip the request saturates
# instead. With a standardized score it never engages below lambda ~ 50.
_EXP_CLIP = 300.0

MODES = ("squared", "abs", "signed")

NONE_KEY = ("none", 0)


class RedundancyScore(NamedTuple):
    """What compute_red returns.

    names:    op order, matching the rows of the matrix it was built from
    raw:      row sums before standardization, in the units of `mode`
    std:      the standardized score -- what reweight consumes
    dropped:  ops whose row was entirely unusable (all-NaN, or gated out)
    mode:     which reduction produced it
    reason:   None when the score is usable; otherwise why it is not, and the
              caller must leave the pdf alone. Never silently degrade -- a no-op
              that looks like a real run is the expensive failure here.
    """

    names: List[str]
    raw: np.ndarray
    std: np.ndarray
    dropped: List[str]
    mode: str
    reason: Optional[str]

    def as_dict(self) -> Dict[str, float]:
        """
        Convert standardized redundancy scores to a mapping keyed by operation name.
        
        Returns:
        	Dict[str, float]: Standardized score for each operation.
        """
        return {name: float(v) for name, v in zip(self.names, self.std)}

    @property
    def usable(self) -> bool:
        """Indicate whether the redundancy score is usable.
        
        Returns:
        	bool: `True` if the score has no unusability reason, `False` otherwise.
        """
        return self.reason is None


def within_op_pairs(names: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Identify paired augmentation directions that belong to the same operation.
    
    Parameters:
        names (Sequence[str]): Augmentation names to inspect.
    
    Returns:
        List[Tuple[int, int]]: Index pairs for matching ``lighter_*/darker_*`` or
            ``*_pos/*_neg`` augmentation names.
    """
    index = {name: i for i, name in enumerate(names)}
    pairs = []
    for name, i in index.items():
        if name.startswith("lighter_"):
            partner = "darker_" + name[len("lighter_") :]
        elif name.endswith("_pos"):
            partner = name[: -len("_pos")] + "_neg"
        else:
            continue
        if partner in index:
            pairs.append((i, index[partner]))
    return pairs


def compute_red(
    r,
    names: Sequence[str],
    mode: str = "squared",
    mask_within_op: bool = True,
    survivor_mask=None,
    min_std: float = 1e-6,
) -> RedundancyScore:
    """
    Compute standardized per-operation redundancy scores from a correlation matrix.
    
    Parameters:
        r: A square correlation matrix.
        names: Operation names corresponding to the matrix rows and columns.
        mode: Reduction applied to correlations: ``"squared"``, ``"abs"``, or
            ``"signed"``.
        mask_within_op: Whether to exclude paired directions of the same operation.
        survivor_mask: Optional boolean matrix identifying cells that contribute to
            the scores.
        min_std: Minimum score standard deviation required for usable results.
    
    Returns:
        RedundancyScore: Raw and standardized scores, dropped operations, and any
            reason the scores are unusable.
    
    Raises:
        ValueError: If the mode, matrix shape, name count, or survivor-mask shape
            is invalid.
    """
    if mode not in MODES:
        raise ValueError(f"unknown red mode {mode!r}, expected one of {MODES}")

    names = list(names)
    matrix = np.asarray(r, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"r must be square, got shape {matrix.shape}")
    if matrix.shape[0] != len(names):
        raise ValueError(
            f"r is {matrix.shape[0]}x{matrix.shape[0]} but {len(names)} names were given"
        )

    n_ops = len(names)
    work = matrix.copy()

    # An op whose whole off-diagonal row is NaN was dropped upstream (constant
    # gradient row -> no correlation defined). Record it before the NaNs are
    # zeroed, otherwise it is indistinguishable from an op that genuinely
    # correlates with nothing.
    off_diagonal = ~np.eye(n_ops, dtype=bool)
    all_nan = np.all(np.isnan(np.where(off_diagonal, work, np.nan)), axis=1)
    dropped = [name for name, is_nan in zip(names, all_nan) if is_nan]

    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(work, 0.0)

    if mask_within_op:
        for i, j in within_op_pairs(names):
            work[i, j] = 0.0
            work[j, i] = 0.0

    if survivor_mask is not None:
        mask = np.asarray(survivor_mask, dtype=bool)
        if mask.shape != work.shape:
            raise ValueError(
                f"survivor_mask is {mask.shape}, expected {work.shape}"
            )
        work = work * mask

    if mode == "squared":
        raw = (work**2).sum(axis=1)
    elif mode == "abs":
        raw = np.abs(work).sum(axis=1)
    else:
        raw = work.sum(axis=1)

    std_dev = float(raw.std())
    reason = None
    if len(dropped) == n_ops:
        reason = "every op was dropped upstream; R carries no usable cell"
    elif not np.any(work):
        reason = (
            "no cell survived masking (within-op exclusion and/or the FDR gate); "
            "red(a) is identically zero"
        )
    elif std_dev < min_std:
        reason = (
            f"red(a) has standard deviation {std_dev:.3g} < {min_std:g}: every op is "
            f"equally redundant, so reweighting would be a no-op"
        )

    standardized = (raw - raw.mean()) / (std_dev + _STD_FLOOR)

    return RedundancyScore(
        names=names,
        raw=raw,
        std=standardized,
        dropped=dropped,
        mode=mode,
        reason=reason,
    )


def ramp_lambda(target: float, progress: float, mode: str = "linear") -> float:
    """
    Scale the reweighting strength according to training progress.
    
    Parameters:
        target (float): Desired lambda value at full progress.
        progress (float): Fraction of training completed.
        mode (str): Scaling mode, either ``"linear"`` or ``"constant"``.
    
    Returns:
        float: The scaled lambda value.
    
    Raises:
        ValueError: If ``mode`` is not ``"linear"`` or ``"constant"``.
    """
    if mode == "constant":
        return float(target)
    if mode == "linear":
        return float(target) * float(np.clip(progress, 0.0, 1.0))
    raise ValueError(f"unknown lambda ramp {mode!r}, expected 'linear' or 'constant'")


class ReweightResult(NamedTuple):
    pdf: Dict
    applied: bool
    reason: Optional[str]
    spread: float  # max/min over the reweighted entries; 1.0 when nothing changed


def reweight(
    pdf_dict: Dict,
    red: Optional[Dict[str, float]],
    lam: float,
    hold_keys: Iterable = (NONE_KEY,),
) -> ReweightResult:
    """
    Reweights augmentation probabilities according to per-operation redundancy scores while preserving the relative probability of each operation's levels.
    
    Parameters:
        pdf_dict (Dict): Probability distribution keyed by operations or operation-level pairs.
        red (Optional[Dict[str, float]]): Per-operation redundancy scores.
        lam (float): Reweighting strength.
        hold_keys (Iterable): Distribution keys whose probabilities remain fixed.
    
    Returns:
        ReweightResult: The reweighted distribution and its application status.
    """
    held = set(hold_keys)

    if lam == 0:
        return ReweightResult(dict(pdf_dict), False, "lambda is 0", 1.0)
    if not red:
        return ReweightResult(dict(pdf_dict), False, "no redundancy score available", 1.0)
    if not pdf_dict:
        return ReweightResult(dict(pdf_dict), False, "pdf is empty", 1.0)

    free_keys = [key for key in pdf_dict if key not in held]
    if not free_keys:
        return ReweightResult(
            dict(pdf_dict), False, "every pdf entry is held fixed", 1.0
        )

    weights = np.array([pdf_dict[key] for key in free_keys], dtype=np.float64)
    free_mass = float(weights.sum())
    if free_mass <= 0:
        return ReweightResult(
            dict(pdf_dict), False, "the reweightable entries carry no mass", 1.0
        )

    scores = np.array(
        [float(red.get(_op_of(key), 0.0)) for key in free_keys], dtype=np.float64
    )
    if not np.any(scores):
        return ReweightResult(
            dict(pdf_dict), False, "no pdf entry has a redundancy score", 1.0
        )

    # Re-centring is mathematically a no-op -- the constant cancels in the
    # normaliser, which is the whole reason standardization is safe -- but it keeps
    # the exponent centred on 0, so the clip below bites symmetrically instead of
    # lopping off one tail of an un-standardized score.
    exponent = np.clip(-float(lam) * (scores - scores.mean()), -_EXP_CLIP, _EXP_CLIP)
    # Shift the top to 0 before exponentiating: exp() then lands in (0, 1] and
    # cannot overflow, and the sum is at least as large as its biggest term, so the
    # renormalisation below cannot underflow the small end either.
    exponent -= exponent.max()
    new_weights = weights * np.exp(exponent)

    total = float(new_weights.sum())
    if not math.isfinite(total) or total <= 0:
        return ReweightResult(
            dict(pdf_dict), False, "reweighting produced a degenerate distribution", 1.0
        )

    # Renormalise to the mass the free entries started with, so the held-fixed
    # entries keep their exact probability and the whole pdf still sums to 1.
    new_weights *= free_mass / total

    out = dict(pdf_dict)
    for key, value in zip(free_keys, new_weights):
        out[key] = float(value)

    positive = new_weights[new_weights > 0]
    spread = float(positive.max() / positive.min()) if positive.size else 1.0

    return ReweightResult(out, True, None, spread)


def _op_of(key):
    """The op name out of a pdf key. Keys are (op, level) tuples, but tolerate a
    bare string so this is usable against an op-keyed dict too."""
    if isinstance(key, tuple):
        return key[0]
    return key


def summarise(score: RedundancyScore, top: int = 5) -> str:
    """
    Create a one-line summary of redundancy scores for training logs.
    
    Parameters:
        score (RedundancyScore): Redundancy scores and associated operation metadata.
        top (int): Maximum number of most and least redundant operations to include.
    
    Returns:
        str: A formatted summary of score statistics and the most and least redundant operations, or the reason the scores are unusable.
    """
    if not score.usable:
        return f"red({score.mode}) unusable: {score.reason}"
    order = np.argsort(score.std)
    least = ", ".join(
        f"{score.names[i]}={score.std[i]:+.2f}" for i in order[:top]
    )
    most = ", ".join(
        f"{score.names[i]}={score.std[i]:+.2f}" for i in order[::-1][:top]
    )
    return (
        f"red({score.mode}) over {len(score.names)} ops "
        f"[raw mean {score.raw.mean():+.2f}, sd {score.raw.std():.2f}] -- "
        f"most redundant: {most} | least: {least}"
    )
