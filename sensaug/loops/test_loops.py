"""Development and inspection loops -- not part of a training run.

None of these are referenced by `train.py`. `VisualizeSampleLoop` and
`DebugAugLoop` compute no metrics at all; they dump sample JPEGs into
`<work_dir>/img_samples` and return `None`. `KIDTestLoop` asserts a
`corruption_dataloader` cfg key that `train.py` never sets.

`DebugAugLoop` is the one with a live caller: `test_robust.py` sets
`cfg.test_cfg.type = "DebugAugLoop"` to eyeball what each perturbation actually
does to an image at each magnitude.

The loops that do run during training live in `train_loops.py`, `sensaug_loop.py`
and `grad_corr_loop.py`.
"""

import os
import cv2
from copy import deepcopy

from mmengine.device import get_device
from mmengine.runner import TestLoop
from mmseg.registry import LOOPS

# Check Pytorch installation
import torch
from torchmetrics.image.kid import KernelInceptionDistance

# Local imports -- NEW_PERTURBATIONS and apply_perturbations_dataloader arrive
# through these, as they always have.
from sensaug.sensitivity_analysis import *  # noqa: F401,F403
from sensaug.runner_utils import *  # noqa: F401,F403

__all__ = [
    "KIDTestLoop",
    "VisualizeSampleLoop",
    "DebugAugLoop",
]


@LOOPS.register_module()
class KIDTestLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        assert "corruption_dataloader" in self.runner.cfg.keys(), (
            "corruption dataloader cfg not found"
        )

        self.kid = None
        self.max_iters = int(len(self.dataloader))
        self.update_kid()
        self.kid.reset()  # reset KID metric obj
        assert self.kid is not None, "KID not initialized"

    def update_kid(self):
        self.kid = KernelInceptionDistance(
            subsets=self.max_iters,
            subset_size=8,
            reset_real_features=False,
            normalize=False,
        ).to(get_device())

        # use the corruption dataloader
        dataloader_cfg = deepcopy(self.runner.cfg.corruption_dataloader)

        diff_rank_seed = self.runner._randomness_cfg.get("diff_rank_seed", False)
        corruption_dataloader = self.runner.build_dataloader(
            dataloader_cfg, seed=self.runner.seed, diff_rank_seed=diff_rank_seed
        )

        # update KID with corruption dataloader samples (e.g., one transform from ImageNet-C)
        for idx, data_batch in enumerate(corruption_dataloader):
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=True)

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        self.kid.reset()  # reset KID metric obj

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)

        total_samples = len(self.dataloader.dataset)
        kid_result = self.kid.compute()

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")

        metrics["kid"] = kid_result

        return metrics


@LOOPS.register_module()
class VisualizeSampleLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.img_savedir = os.path.join(runner.work_dir, "img_samples")
        os.makedirs(self.img_savedir, exist_ok=True)

    def run(self) -> dict:
        """Launch test."""

        max_iter = 1

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            if idx >= max_iter:
                break

            batchsize = len(data_batch["inputs"])
            for i in range(batchsize):
                sample_n = idx * batchsize + i
                filename = os.path.join(self.img_savedir, f"sample_{sample_n}.jpg")
                cv2.imwrite(filename, data_batch["inputs"][i].permute(1, 2, 0).numpy())
                print(f"saved to: {filename}")
                break
            # self.run_iter(idx, data_batch)

        # total_samples = int(max_iter * self.dataloader.batch_size)

        # compute metrics
        # metrics = self.evaluator.evaluate(total_samples)
        # self.runner.call_hook('after_test_epoch', metrics=metrics)
        # self.runner.call_hook('after_test')

        # return metrics


@LOOPS.register_module()
class DebugAugLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.img_savedir = os.path.join(runner.work_dir, "img_samples")
        os.makedirs(self.img_savedir, exist_ok=True)

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        perturbation_list = list(NEW_PERTURBATIONS.items())  # noqa: F405

        magnitudes = [0.25, 0.50, 0.75, 1.0]
        sample = None

        for pname, _ in perturbation_list:
            for m in magnitudes:
                apply_perturbations_dataloader(  # noqa: F405
                    self.runner, train=False, perturb_levels={pname: m}
                )
                print(f"== pname: {pname} || magnitude: {m}")

                for idx, data_batch in enumerate(self.dataloader):
                    sample = data_batch
                    break

                filename = os.path.join(self.img_savedir, f"sample_{pname}_{m}.jpg")
                cv2.imwrite(filename, sample["inputs"][0].permute(1, 2, 0).numpy())
                print(f"saved to: {filename}")
