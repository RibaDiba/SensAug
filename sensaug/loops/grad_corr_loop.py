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
"""

import json

from mmengine.logging import print_log
from mmseg.registry import LOOPS
from mmengine.dist import is_main_process

from sensaug.hooks.grad_hook import training_progress
from sensaug.redundancy import ramp_lambda, reweight

from .sensaug_loop import RobustValLoop

__all__ = ["GradCorrValLoop"]


@LOOPS.register_module()
class GradCorrValLoop(RobustValLoop):
    """RobustValLoop plus Lever 3's redundancy down-weighting of the training pdf.

    `q(a) ~ pdf_old(a) * exp(-lambda * red(a))` -- the closed-form solution to
    minimising `KL(q || pdf_old)` subject to a budget on `sum q(a)*red(a)`. The
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
        perturbation_set: str = "new",
        corr_lambda: float = 0.0,
        corr_lambda_ramp: str = "linear",
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
        """
        if not self.corr_lambda:
            return pdf_dict

        published = getattr(self.runner, "corr_redundancy", None)
        if not published:
            return pdf_dict

        lam = ramp_lambda(
            self.corr_lambda,
            training_progress(self.runner),
            mode=self.corr_lambda_ramp,
        )
        result = reweight(pdf_dict, published.get("red"), lam)

        if is_main_process():
            if result.applied:
                print_log(
                    f"[redundancy] round {self.n_rounds}: reweighted the pdf at "
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
                    f"[redundancy] round {self.n_rounds}: pdf NOT reweighted -- "
                    f"{result.reason}",
                    logger="current",
                    level=30,  # WARNING
                )

        return result.pdf
