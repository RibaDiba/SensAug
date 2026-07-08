"""
Hook that computes the cross-correlation matrix R between the differentiable
augmentations, from the per-op gradient traces collected by CollectGradientHook.

This is a passive consumer: CollectGradientHook accumulates, every probe, one
``d loss / d magnitude`` scalar per differentiable op into
``runner.aug_grad_buffer`` (aligned columns -- all ops measured on the same
batch). At the end of each RobustValLoop round (``after_val_epoch``, i.e. right
after that round's sensitivity-analysis sweep), this hook stacks the buffer into
a matrix D_grad (A x T), computes R = corrcoef(D_grad), logs it, and clears the
buffer window.

Scope: R covers the differentiable ops only (the ones with gradients); pruning
based on R is future work -- ``prune_augmentations`` is a stub.
"""

import os
import json

import numpy as np

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS


@HOOKS.register_module()
class PerturbationSensitivityAnalysisHookWithGradients(Hook):
    """Builds and logs the augmentation gradient cross-correlation matrix R.

    Consumes ``runner.aug_grad_buffer`` (populated by CollectGradientHook); no
    effect unless that hook is also registered and has accumulated >= 2 probes.
    """

    def __init__(self) -> None:
        self.corr_log_path = None
        self.names = list(DIFFERENTIABLE_PERTURBATIONS)

    def after_val_epoch(self, runner: Runner, metrics=None) -> None:
        if self.corr_log_path is None:
            self.corr_log_path = os.path.join(
                runner.cfg.work_dir, "corr_matrix_log.txt"
            )

        buffer = getattr(runner, "aug_grad_buffer", None)
        if buffer is None:
            return

        # Need at least 2 aligned probes per op for a correlation.
        lengths = [len(buffer[name]) for name in self.names]
        if min(lengths) < 2:
            return

        # Only rank 0 holds meaningful probe traces / does the logging. R is a
        # logged diagnostic for now (no action taken), so no cross-rank reduce.
        if is_main_process():
            n = min(lengths)  # truncate so columns align
            d_grad = np.array([buffer[name][:n] for name in self.names])

            r = self.calculate_cross_corelation(d_grad)

            round_idx = getattr(getattr(runner, "val_loop", None), "n_rounds", None)
            record = {
                "round": round_idx,
                "names": self.names,
                "R": r.tolist(),
            }
            with open(self.corr_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            print_log(
                f"Augmentation gradient cross-correlation matrix computed "
                f"(round {round_idx}, {n} probes)",
                logger="current",
            )

            self.prune_augmentations(r)

        # Clear the window on every rank so each round's R reflects only the
        # gradients collected since the previous round.
        for name in buffer:
            buffer[name].clear()

    def calculate_cross_corelation(self, d_grad: np.ndarray) -> np.ndarray:
        """Pearson cross-correlation across the augmentation rows of D_grad.

        Args:
            d_grad (np.ndarray): shape (A, T) -- A differentiable ops x T probes.

        Returns:
            np.ndarray: symmetric (A, A) correlation matrix in [-1, 1].
        """
        return np.corrcoef(d_grad)

    def prune_augmentations(self, r: np.ndarray) -> None:
        """Prune redundant augmentations given the correlation matrix R.

        Not implemented yet -- the pruning rule is still being researched. For
        now this hook only computes and logs R; nothing is applied back to the
        training distribution.
        """
        pass
