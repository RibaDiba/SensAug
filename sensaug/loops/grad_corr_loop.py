"""The gradient cross-correlation pipeline's val loop: `GradCorrValLoop`.

This is the loop `--aug-type=grad_corr` runs. It is a strict superset of
`RobustValLoop` (`sensaug_loop.py`): identical SA machinery -- same SA curve, same
round-eval, same three pdf generators -- plus Lever 3, which down-weights the
augmentations the correlation matrix R found redundant with the rest of the bank.

Note what is NOT here. R itself is built in `sensaug/hooks/` --
`CollectGradientHook` sweeps for `d loss / d magnitude` and
`PerturbationSensitivityAnalysisHookWithGradients` correlates the sweep, both off
`corr_interval` in `after_train_iter`. This loop runs on `round_interval` and
never has to agree with them; it just reads whatever score is current. The two
pipelines share no clock and no hook point.

So the only thing this class adds is the consumer side of that handoff:
`_apply_redundancy_reweighting`, the one extension point the base declares.

**The down-weighting method is pluggable** -- see `DOWNWEIGHT_METHODS` below.
`red(a)` -> pdf is a modelling choice, not the mechanism, and it is the third axis
of this pipeline: `--corr-red-mode` picks how a row of R reduces to a scalar,
`--corr-lambda` picks how hard the result pushes, and `--corr-downweight-method`
picks the functional form of the push itself. New arms are added to one dict in
this file rather than by editing the call path.
"""

import json
import math

import numpy as np

from mmengine.logging import print_log
from mmseg.registry import LOOPS
from mmengine.dist import is_main_process

from sensaug.hooks.grad_hook import training_progress
from sensaug.redundancy import (
    NONE_KEY,
    ReweightResult,
    ramp_lambda,
    reweight,
    within_op_pairs,
)

from .sensaug_loop import RobustValLoop

__all__ = [
    "GradCorrValLoop",
    "DOWNWEIGHT_METHODS",
    "HARD_PRUNING_METHODS",
    "resolve_downweight_method",
]


# --------------------------------------------------------------------------- #
# The down-weighting methods.
#
# Every method has the same signature and returns the same type::
#
#     fn(pdf_dict: dict, published: dict, lam: float) -> ReweightResult
#
# `published` is the WHOLE record the correlation hook put on
# `runner.corr_redundancy`, not just its "red" field. Today only that field is
# read, but a method built on the structure of R rather than on a per-op summary
# of it (clustering the ops by |R|, projecting onto R's leading eigenvector) needs
# more than the row sums, and passing the record whole means adding one costs a
# function here instead of a signature change through every arm.
#
# `lam` arrives already ramped (see `_apply_redundancy_reweighting`), so every
# method inherits `--corr-lambda-ramp` for free and none of them re-implements it.
# `lam == 0` never reaches a method at all -- the caller short-circuits, which is
# what keeps the control arm bit-identical rather than approximately identical.
#
# Returning `redundancy.ReweightResult` is the contract that matters: the
# applied / reason / spread logging below is written against it, so a method that
# does not move the pdf -- whether by design (`none`) or because it could not (a
# degenerate score, an empty pdf) -- reports WHY in the same shape as every other
# method, and a silent no-op cannot masquerade as a real arm.
#
# A method DECLINES, it does not raise: the call happens mid-training on every
# rank, so an exception here kills the job several hours in.
# --------------------------------------------------------------------------- #


def _downweight_none(pdf_dict: dict, published: dict, lam: float):
    """No down-weighting: the pdf is used exactly as the SA loop generated it.

    Ignores both the published score and lambda, deliberately -- this is the arm
    where R is measured, logged and simply not fed back into training, so it is
    the baseline every other method is read against.

    A named arm rather than "the flag was left off": `--aug-type=grad_corr`
    requires a method, so a run that does not down-weight says so in its own
    launch command and in its own logs, instead of being inferred from the absence
    of something.
    """
    return ReweightResult(
        dict(pdf_dict), False, "method 'none': down-weighting is off", 1.0
    )


def _downweight_soft_weighting(pdf_dict: dict, published: dict, lam: float):
    """Max-entropy tilt: `q(a) ~ pdf_old(a) * exp(-lambda * red(a))`.

    The closed-form solution to minimising `KL(q || pdf_old)` subject to a budget
    on `sum_a q(a) red(a)`, with lambda the Lagrange multiplier on that budget --
    a derivation rather than a heuristic. Soft by construction: exp() is strictly
    positive, so an op's probability is pushed down but structurally cannot reach
    zero, which at the correlation sizes actually observed here (mean |r| of
    0.11-0.22) is the strongest claim the measurement supports.

    A thin adapter on purpose. The numerics live in `sensaug/redundancy.py`, which
    is deliberately free of mmseg and torch so `scripts/calibrate_lambda.py` can
    sweep lambda against a logged `corr_matrix_log.json` with no GPU and no
    training run. Methods added here later have no such offline consumer and can
    simply be written out in full.
    """
    return reweight(pdf_dict, published.get("red"), lam)


# --------------------------------------------------------------------------- #
# mRMR -- the hard-pruning arm, and the only method here that deletes.
#
# Everything above is soft by construction: exp() is strictly positive, so an op
# is pushed down and structurally cannot reach zero. mRMR makes the opposite
# modelling claim -- that a sufficiently redundant op should not be sampled AT ALL
# -- and it is a stronger claim than R currently supports (at the correlation
# sizes actually observed here, mean |r| of 0.11-0.22, deletion is not implied by
# the measurement). It is a legitimate arm to run and to report; it is not the one
# to reach for by default, and a result from it should be read next to a
# soft-weighting run at the same lambda rather than on its own.
#
# The geometric caveat in CLAUDE.md bites harder here for the same reason: the 10
# geometric ops' R is contaminated by image-label misalignment, which makes them
# look LEAST redundant, so mRMR will preferentially keep them and prune
# photometric ops. Pair any reportable mRMR run with --photometric-only until
# warp_image_and_label is wired into the probe.
# --------------------------------------------------------------------------- #

#: Guards the two standardizations below when the spread is degenerate. A uniform
#: pdf (generate_pdf_new before the SA curve exists, and --uniform for the whole
#: run) makes the relevance spread exactly 0, which is not an error here -- it
#: just means the ranking falls through to pure minimum-redundancy selection.
_MRMR_STD_FLOOR = 1e-8


def _op_of(key):
    """The op name out of a pdf key. Keys are (op, level); a bare string is
    tolerated so this works against an op-keyed dict too."""
    return key[0] if isinstance(key, tuple) else key


def _finite_mean(values):
    """Mean over the finite entries, NaN when there are none.

    np.nanmean would do it but warns on an all-NaN slice, and all-NaN slices are
    expected here rather than exceptional: an op whose partners all failed the FDR
    gate has no measured redundancy against the selected set.
    """
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _as_matrix(value):
    """Coerce a published R into a float matrix, tolerating the JSON round trip.

    In-process this is the ndarray the correlation hook built, NaN and all. Read
    back out of a log it is nested lists in which `_jsonable` has already turned
    every non-finite cell into `None` -- and `np.asarray(..., dtype=float)` raises
    on those rather than reading them as missing.
    """
    return np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in value],
        dtype=np.float64,
    )


def _pairwise_redundancy(published):
    """`(names, W, reason)` -- the pairwise redundancy mRMR penalises.

    `W[i, j]` is how redundant op i is with op j, and NaN wherever the pair carries
    no usable evidence: the diagonal, a within-op pair, an op R dropped upstream,
    or a cell that did not survive the FDR gate. `reason` is non-None exactly when
    the matrix cannot be used, and the caller must then leave the pdf alone.

    Built from the same cells `redundancy.compute_red` sums into `red(a)`, under
    the same two masks and the same `--corr-red-mode` reduction. That is
    deliberate: the soft and hard arms should differ in what they DO with R, not
    in what they think R says.
    """
    names = published.get("names")
    matrix = published.get("r")
    if not names or matrix is None:
        return None, None, (
            "no correlation matrix was published: mRMR ranks ops against each "
            "other, so it needs the pairwise R and cannot run off the per-op row "
            "sums alone"
        )

    names = list(names)
    try:
        work = _as_matrix(matrix)
    except (TypeError, ValueError) as exc:
        return None, None, f"the published R could not be read as a matrix ({exc})"

    if work.ndim != 2 or work.shape != (len(names), len(names)):
        return None, None, (
            f"the published R is {work.shape} but {len(names)} op names came with "
            f"it; refusing to guess the axis order"
        )

    mode = published.get("mode", "squared")
    if mode == "squared":
        work = work**2
    elif mode == "abs":
        work = np.abs(work)
    elif mode == "signed":
        # Same asymmetry --corr-red-mode already documents: under 'signed' an
        # anti-correlated pair scores negative, i.e. it PROTECTS both ops from
        # being pruned rather than pruning either.
        work = work.copy()
    else:
        return None, None, (
            f"the published record was scored with an unknown red mode {mode!r}"
        )

    np.fill_diagonal(work, np.nan)
    if published.get("mask_within_op", True):
        for i, j in within_op_pairs(names):
            work[i, j] = np.nan
            work[j, i] = np.nan

    survives = published.get("survives")
    if survives is not None:
        mask = np.asarray(survives, dtype=bool)
        if mask.shape != work.shape:
            return None, None, (
                f"the published FDR survivor mask is {mask.shape}, expected "
                f"{work.shape}"
            )
        # A cell that did not survive multiplicity correction is not evidence of
        # redundancy, and here that evidence would delete an augmentation. Drop it
        # rather than shrink it.
        work = np.where(mask, work, np.nan)

    return names, work, None


def _downweight_mrmr(pdf_dict: dict, published: dict, lam: float):
    """mRMR hard pruning: rank the ops by minimum-Redundancy Maximum-Relevance,
    keep a lambda-sized prefix of that ranking, and set every other op's
    probability to exactly ZERO.

    Two halves, and lambda drives only the second one:

    * **The ranking** is textbook greedy mRMR. Seed with the most relevant op,
      then repeatedly take the op maximising `rel(a) - red(a | S)` against the
      already-selected set S. Both terms are standardized across ops so the
      difference is dimensionless and neither term can dominate by unit choice
      alone -- which is what lets the objective stay lambda-free, unlike the
      max-entropy tilt where lambda has to bridge a log-probability against a
      correlation.

      **Relevance is the SA loop's own pdf**, summed over an op's magnitude
      levels. That is already the pipeline's statement of which perturbations the
      model is currently worst at, so "maximum relevance" needs no second signal
      and inherits `--uniform` / `--weighted-augs` for free. When the pdf is
      uniform (before the SA curve exists, or for a whole `--uniform` run) the
      relevance term is flat and the ranking degenerates gracefully into pure
      minimum-redundancy selection rather than into arbitrary order.

    * **The budget** is `ceil(A / (1 + lambda))` of the A measured ops. lambda=0
      keeps everything -- continuous with the caller's short-circuit rather than
      discontinuous at it -- 0.25 keeps ~80%, 0.5 ~67%, 1.0 half, 2.0 a third. So
      lambda still reads as "how hard redundancy pushes"; on this arm it pushes ops
      out of the bank instead of down the pdf. It composes with
      `--corr-lambda-ramp` the same way, which means early rounds (where R
      describes a model that barely discriminates between augmentations yet) prune
      little or nothing and the bank narrows as the measurement earns it.

    Three things worth knowing before running it:

    * **The prune is not latched.** It is re-derived from the current pdf and the
      current R every round, so an op pruned at one round can return at the next.
      That is the intended behaviour -- the alternative commits the run to a
      decision made off the earliest and least trustworthy R.
    * **An op with no usable row in R is exempt**, never pruned. Deleting an
      augmentation on the strength of a measurement that does not exist is the one
      failure mode here that cannot be argued for, so absence of evidence buys
      immunity. A heavily FDR-gated R therefore prunes little, and says so.
    * **`("none", 0)` is held fixed**, as on every arm: this changes which
      augmentation is sampled, never how often augmentation happens.
    """
    if not pdf_dict:
        return ReweightResult(dict(pdf_dict), False, "pdf is empty", 1.0)

    names, work, reason = _pairwise_redundancy(published)
    if reason is not None:
        return ReweightResult(dict(pdf_dict), False, reason, 1.0)

    held = {NONE_KEY}
    relevance = {}
    for key, prob in pdf_dict.items():
        if key in held:
            continue
        op = _op_of(key)
        relevance[op] = relevance.get(op, 0.0) + float(prob)
    if not relevance:
        return ReweightResult(
            dict(pdf_dict), False, "every pdf entry is held fixed", 1.0
        )

    index = {name: i for i, name in enumerate(names)}
    measured = [
        op
        for op in sorted(relevance)
        if op in index and np.isfinite(work[index[op]]).any()
    ]
    exempt = [op for op in sorted(relevance) if op not in measured]
    if not measured:
        return ReweightResult(
            dict(pdf_dict),
            False,
            "no op in the pdf has a usable row in R (vocabulary mismatch, or every "
            "cell was masked out); there is nothing to rank",
            1.0,
        )

    budget = max(1, math.ceil(len(measured) / (1.0 + float(lam))))
    if budget >= len(measured):
        return ReweightResult(
            dict(pdf_dict),
            False,
            f"lambda={lam:.3g} gives a budget of {budget} of {len(measured)} "
            f"measured ops, so nothing is pruned",
            1.0,
        )

    idx = np.array([index[op] for op in measured])
    sub = work[np.ix_(idx, idx)]

    rel = np.array([relevance[op] for op in measured], dtype=np.float64)
    rel_z = (rel - rel.mean()) / (rel.std() + _MRMR_STD_FLOOR)

    cells = sub[np.isfinite(sub)]
    red_mu = float(cells.mean()) if cells.size else 0.0
    red_sd = float(cells.std()) if cells.size else 0.0

    # Per-op redundancy against the whole bank, used only to break ties. Ties on
    # relevance are the norm here rather than the exception, and breaking them on
    # the globally least-redundant op keeps the seed of the greedy chain
    # meaningful -- and identical on every rank -- instead of falling back to
    # alphabetical order.
    global_red = np.array([_finite_mean(row) for row in sub], dtype=np.float64)
    global_red = np.where(np.isnan(global_red), red_mu, global_red)

    selected = []
    remaining = list(range(len(measured)))
    while len(selected) < budget and remaining:
        if selected:
            penalty = np.array(
                [_finite_mean(sub[i, selected]) for i in remaining], dtype=np.float64
            )
            # No measured cell against the selected set means no evidence either
            # way, which is the average, not zero redundancy.
            penalty = np.where(np.isnan(penalty), red_mu, penalty)
            scores = rel_z[remaining] - (penalty - red_mu) / (red_sd + _MRMR_STD_FLOOR)
        else:
            scores = rel_z[remaining]
        # lexsort takes its PRIMARY key last: score descending, then redundancy
        # ascending.
        pick = int(np.lexsort((global_red[remaining], -scores))[0])
        selected.append(remaining.pop(pick))

    survivors = {measured[i] for i in selected}
    survivors.update(exempt)
    pruned = sorted(op for op in relevance if op not in survivors)
    if not pruned:
        return ReweightResult(
            dict(pdf_dict), False, "the ranking kept every op; nothing was pruned", 1.0
        )

    free_keys = [key for key in pdf_dict if key not in held]
    free_mass = float(sum(pdf_dict[key] for key in free_keys))
    survivor_mass = float(
        sum(pdf_dict[key] for key in free_keys if _op_of(key) in survivors)
    )
    if free_mass <= 0 or survivor_mass <= 0:
        return ReweightResult(
            dict(pdf_dict), False, "the surviving ops carry no pdf mass", 1.0
        )

    # The pruned mass goes to the survivors in proportion to what they already
    # held, so an op's beta-binomial shape over its magnitude levels is preserved
    # exactly and only the mass allotted BETWEEN ops moves -- the same invariant
    # redundancy.reweight maintains.
    scale = free_mass / survivor_mass
    out = dict(pdf_dict)
    for key in free_keys:
        out[key] = float(pdf_dict[key]) * scale if _op_of(key) in survivors else 0.0

    # Over the FREE entries only, and only the surviving ones -- the same quantity
    # redundancy.reweight reports, so the "max/min over the perturbation mass"
    # line in the log means the same thing on both arms. Including the held-fixed
    # ("none", 0) mass would make it a different number, and including the zeros
    # would make it infinite.
    positive = np.array(
        [out[key] for key in free_keys if out[key] > 0], dtype=np.float64
    )
    spread = float(positive.max() / positive.min()) if positive.size else 1.0

    if is_main_process():
        # The zeros are visible in the pdf the caller prints, but not WHICH ops
        # they are or why there are that many of them, and on this arm that is the
        # single most important line in the round.
        exempt_note = f", {len(exempt)} exempt (no usable row in R)" if exempt else ""
        print_log(
            f"[redundancy] mRMR pruned {len(pruned)}/{len(relevance)} ops at "
            f"lambda={lam:.3g} (budget {budget} of {len(measured)} measured"
            f"{exempt_note}): {', '.join(pruned)}",
            logger="current",
        )

    return ReweightResult(out, True, None, spread)


#: name -> method. THIS is the list to add to: write the function above, add one
#: line here, done. train.py builds its argparse `choices` from these keys and
#: `resolve_downweight_method` validates against them, so the flag, the error
#: message and the dispatch table cannot drift apart.
DOWNWEIGHT_METHODS = {
    "none": _downweight_none,
    "soft-weighting": _downweight_soft_weighting,
    "mRMR": _downweight_mrmr,
}

#: The arms allowed to drive an op's probability to exactly zero.
#:
#: Softness is otherwise a contract, not a coincidence: `tests/` asserts it over
#: every registered method that is NOT named here, so a method added later is held
#: to "soft, never zero" unless its author deliberately opts out by adding it to
#: this set. Deletion should cost a line that someone has to write on purpose.
HARD_PRUNING_METHODS = frozenset({"mRMR"})


def resolve_downweight_method(name):
    """Look up a method by name, or raise `ValueError` naming the valid ones.

    There is no default, and `None` is an error rather than a fallback to "none"
    -- "do not down-weight" is a choice someone made, and it has to look like one.
    An unnamed arm is the failure mode this whole flag exists to prevent: the
    method is not recoverable from the checkpoint, the logs or the work_dir name,
    so a run that did not state it is a run whose results cannot be attributed
    later. `train.py` enforces the same rule at the CLI, and this enforces it for
    every other construction path (a hand-written config, a test).
    """
    try:
        return DOWNWEIGHT_METHODS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown redundancy down-weighting method {name!r}; expected one of "
            f"{sorted(DOWNWEIGHT_METHODS)}"
        ) from None


@LOOPS.register_module()
class GradCorrValLoop(RobustValLoop):
    """RobustValLoop plus Lever 3's redundancy down-weighting of the training pdf.

    Which down-weighting happens is `--corr-downweight-method`'s to pick, out of
    `DOWNWEIGHT_METHODS` above -- `none` leaves the pdf alone, `soft-weighting` is
    the max-entropy tilt `q(a) ~ pdf_old(a) * exp(-lambda * red(a))`, and `mRMR`
    hard-prunes the redundant ops to probability zero. This class is only the
    wiring; the numerics live in `sensaug/redundancy.py` (soft-weighting, kept
    mmseg-free so `scripts/calibrate_lambda.py` can sweep it offline) and above in
    this file (mRMR, which has no offline consumer).
    """

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio=0.5,
        sa_curve_path=None,
        uniform=False,
        descending_MA=False,
        remove_H=False,
        warmup_rounds: int = 4,
        random_aug: bool = False,
        geometric_only: bool = False,
        photometric_only: bool = False,
        weighted_augs: bool = False,
        perturbation_set: str = "legacy20",
        corr_lambda: float = 0.0,
        corr_lambda_ramp: str = "linear",
        corr_downweight_method: str = None,
        fp16: bool = False,
    ) -> None:
        super().__init__(
            runner,
            dataloader,
            evaluator,
            ratio=ratio,
            sa_curve_path=sa_curve_path,
            uniform=uniform,
            descending_MA=descending_MA,
            remove_H=remove_H,
            warmup_rounds=warmup_rounds,
            random_aug=random_aug,
            geometric_only=geometric_only,
            photometric_only=photometric_only,
            weighted_augs=weighted_augs,
            perturbation_set=perturbation_set,
            fp16=fp16,
        )

        # Redundancy down-weighting strength. 0 means the pdf is returned exactly
        # as generated -- bit-identical, not merely close -- which is what makes it
        # usable as the control arm.
        self.corr_lambda = corr_lambda
        self.corr_lambda_ramp = corr_lambda_ramp

        # Resolved HERE, not at the first round that reweights. train.py touches
        # `runner.val_loop` immediately after Runner.from_cfg, so an unknown (or
        # absent) method name fails seconds into the job instead of at round 4 of
        # a multi-hour run, on a compute node, with the walltime already spent.
        self.corr_downweight_method = corr_downweight_method
        self._downweight = resolve_downweight_method(corr_downweight_method)

    def _apply_redundancy_reweighting(self, pdf_dict: dict) -> dict:
        """Down-weight ops the correlation pipeline found redundant with the rest
        of the bank.

        Reads `runner.corr_redundancy`, published by
        PerturbationSensitivityAnalysisHookWithGradients.prune_augmentations. That
        hook fires on `corr_interval` and this loop on `round_interval`, so what is
        read here is simply the latest score -- there is none at all until the first
        emission, and the pdf is returned untouched until then.

        Called by all three pdf generators, so `--corr-lambda` composes with
        `--uniform` and `--weighted-augs` rather than silently applying to only one
        of them.

        The functional form of the down-weighting is `--corr-downweight-method`'s
        to choose; everything around it -- when to act at all, the lambda ramp, and
        the logging -- is shared, so the arms differ in exactly one thing.
        """
        if not self.corr_lambda:
            return pdf_dict

        published = getattr(self.runner, "corr_redundancy", None)
        if not published:
            return pdf_dict

        # Ramped before dispatch: every method inherits --corr-lambda-ramp and
        # none of them re-implements it.
        lam = ramp_lambda(
            self.corr_lambda,
            training_progress(self.runner),
            mode=self.corr_lambda_ramp,
        )
        result = self._downweight(pdf_dict, published, lam)

        if is_main_process():
            if result.applied:
                print_log(
                    f"[redundancy] round {self.n_rounds}: reweighted the pdf by "
                    f"'{self.corr_downweight_method}' at "
                    f"lambda={lam:.3f} (target {self.corr_lambda:g}, "
                    f"{self.corr_lambda_ramp} ramp) off the "
                    f"{published.get('checkpoint')} checkpoint's "
                    f"{published.get('mode')} score -- max/min over the perturbation "
                    f"mass is now {result.spread:.1f}x",
                    logger="current",
                )
                # The caller already printed the pdf as generated; this is the
                # after half of the pair, which is what you want side by side the
                # first time a result looks strange.
                print("Perturbation PDF after redundancy reweighting:")
                print(
                    json.dumps(
                        {str(k): v for k, v in result.pdf.items()},
                        indent=4,
                        sort_keys=True,
                    )
                )
            else:
                # Never silently. A reweighting that quietly does nothing looks
                # exactly like a real experimental arm in the results.
                print_log(
                    f"[redundancy] round {self.n_rounds}: pdf NOT reweighted by "
                    f"'{self.corr_downweight_method}' -- {result.reason}",
                    logger="current",
                    level=30,  # WARNING
                )

        return result.pdf
