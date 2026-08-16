"""Loops that run as part of a real training job.

`RobustIterBasedTrainLoop` is the train loop itself (`train.py` sets
`cfg.train_cfg.type` to it); `SubsetTestLoop` is the eval loop it is paired with
(`cfg.test_cfg.type`, and `sensitivity_analysis.py` asserts on that exact name).
`SubsetValLoop` is the plain subset-eval counterpart.

The SA and gradient cross-correlation val loops live in `sensaug_loop.py` and
`grad_corr_loop.py`; the inspection-only loops live in `test_loops.py`.
"""

import logging

from mmengine.logging import print_log
from mmengine.device import get_device
from mmengine.runner import ValLoop, TestLoop, IterBasedTrainLoop
from mmseg.registry import LOOPS

# Check Pytorch installation
import torch
from torchmetrics.image.kid import KernelInceptionDistance

__all__ = [
    "RobustIterBasedTrainLoop",
    "SubsetValLoop",
    "SubsetTestLoop",
]


@LOOPS.register_module()
class RobustIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader,
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 1000,
        init_sa=False,
        dynamic_intervals=None,
    ) -> None:
        super().__init__(
            runner, dataloader, max_iters, val_begin, val_interval, dynamic_intervals
        )

        self.init_sa = init_sa

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0 and self._iter < self._max_iters:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data "
                "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        if self.init_sa:
            print_log("Initializing sensitivity analysis curve", logger="current")
            self.runner.val_loop.run()
        else:
            print_log(
                f"Not initializing sensitivity analysis curve. Training will resume at iteration {self._iter}. ",
                logger="current",
            )

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (
                    self._iter % self.val_interval == 0 or self._iter == self._max_iters
                )
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model


@LOOPS.register_module()
class SubsetValLoop(ValLoop):
    """Custom loop for test, uses a subset."""

    def __init__(
        self, runner, dataloader, evaluator, ratio=0.5, fp16: bool = False
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.ratio = ratio
        self.dataloader_iter = iter(self.dataloader)

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        max_iter = int(len(self.dataloader) * self.ratio)

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics


@LOOPS.register_module()
class SubsetTestLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(
        self, runner, dataloader, evaluator, ratio=0.5, fp16: bool = False
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.ratio = ratio
        # self.root_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kid = None
        self.update_kid()

    def update_kid(self):
        max_iter = int(len(self.dataloader) * self.ratio)
        # max_iter = int(len(self.dataloader))

        self.kid = KernelInceptionDistance(
            subsets=max_iter,
            subset_size=4,
            reset_real_features=False,
            normalize=False,
        ).to(get_device())

        # set test performance on
        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=True)
            idx += 1

        self.kid.reset()  # reset KID metric obj

    def run(self) -> dict:
        """Launch test."""

        max_iter = int(len(self.dataloader) * self.ratio)

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        self.kid.reset()  # reset KID metric obj

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)
        kid_result = self.kid.compute()

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")

        metrics["kid"] = kid_result

        return metrics

    def run_eval_only(self) -> dict:
        """Launch test."""

        max_iter = int(len(self.dataloader) * self.ratio)
        self.runner.model.eval()
        self.kid.reset()  # reset KID metric obj

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)
        kid_result = self.kid.compute()
        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        metrics["kid"] = kid_result

        return metrics
