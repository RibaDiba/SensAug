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

from mmengine.logging import print_log
from mmseg.registry import LOOPS
from mmengine.dist import is_main_process

from sensaug.hooks.grad_hook import training_progress
from sensaug.redundancy import ReweightResult, ramp_lambda, reweight

from .sensaug_loop import RobustValLoop

__all__ = [
    "GradCorrValLoop",
    "DOWNWEIGHT_METHODS",
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


#: name -> method. THIS is the list to add to: write the function above, add one
#: line here, done. train.py builds its argparse `choices` from these keys and
#: `resolve_downweight_method` validates against them, so the flag, the error
#: message and the dispatch table cannot drift apart.
DOWNWEIGHT_METHODS = {
    "none": _downweight_none,
    "soft-weighting": _downweight_soft_weighting,
}


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
    `DOWNWEIGHT_METHODS` above -- `soft-weighting` is the max-entropy tilt
    `q(a) ~ pdf_old(a) * exp(-lambda * red(a))`, `none` leaves the pdf alone. The
    mechanism itself is pure numpy in `sensaug/redundancy.py`; this class is only
    the wiring.
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
