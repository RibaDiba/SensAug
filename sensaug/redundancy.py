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
        """The standardized score keyed by op name, which is how reweight and the
        log both want it."""
        return {name: float(v) for name, v in zip(self.names, self.std)}

    @property
    def usable(self) -> bool:
        return self.reason is None


def within_op_pairs(names: Sequence[str]) -> List[Tuple[int, int]]:
    """Index pairs for the two halves of a single op, e.g. lighter_S/darker_S and
    sharpness_pos/sharpness_neg.

    Excluded from red(a) because they measure a parameterization convention, not
    redundancy between two augmentations someone might have chosen independently.
    The two directions of one op are the same structural relationship whichever
    sign it comes out with, and under the `signed` reduction a strongly negative
    within-op cell would hand that op the strongest protection in the matrix --
    exactly backwards.

    Derived from the names rather than hard-coded so it stays correct for both the
    14-op vocabulary the earlier runs logged and the 32-op one (6 pairs and 15
    pairs respectively).
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
    """Per-op redundancy score from the correlation matrix.

    r:             (A, A) correlation matrix. NaN is expected -- `correlate` leaves
                   dropped rows and columns NaN, and the json log writes them null.
    survivor_mask: (A, A) bool from fdr_correct. Cells that did not survive
                   multiplicity correction contribute nothing. Standard hygiene at
                   496 simultaneous tests, and it is also what keeps a `signed`
                   variant defensible if the closure question ever resolves toward
                   the constraint binding: a survivor subset sum stays unconstrained
                   even where the full row sum would not.

    `squared` is the default because it is closure-proof and does not depend on the
    sign convention of any individual op. `abs` and `signed` exist as ablation arms.
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
    """Scale lambda by training progress.

    R measured early describes a model that barely discriminates between
    augmentations yet, so applying full strength from step 0 acts hardest on the
    least trustworthy measurement. `progress` is the fraction of training elapsed,
    the same quantity grad_hook.training_progress() reports.
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
    """Max-entropy reweighting of a (op, level)-keyed pdf by a per-op score.

    red(a) is per-op, so it scales every level of that op by the same factor: the
    beta-binomial shape generate_pdf_new gives an op's levels is preserved exactly,
    and only the mass allotted BETWEEN ops moves. Ops missing from `red` -- dropped
    upstream, or simply not in the vocabulary R was measured over -- score 0, which
    under standardization is the mean, i.e. no adjustment either way.

    lambda=0 returns the input unchanged and bit-identical, not merely close. That
    arm has to reproduce the baseline exactly for any comparison against it to mean
    anything, so it short-circuits rather than multiplying by exp(0) and
    renormalising through float error.
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
    """One-line human summary for the training log: which ops this would push down
    hardest and which it would leave alone."""
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
