"""
Hook for probing and storing the gradient of the training loss with respect to
each differentiable augmentation's magnitude.

The differentiable augmentation ops (sensaug.dataset.differentiable_augmentations)
are autograd-compatible but are NOT wired into the training step -- there is no
persistent magnitude parameter whose ``.grad`` we could read after a real
``train_step`` (mmengine's OptimWrapper already ran ``zero_grad`` by the time
``after_train_iter`` fires). So this hook *produces* the gradient itself: every
``probe_interval`` iters it takes a clean batch, applies each differentiable op
at a fixed reference magnitude, runs a forward pass, and reads
``d loss / d magnitude`` via ``torch.autograd.grad`` (which never populates model
parameter ``.grad``, so the real training step is untouched).

The per-op scalar gradients are accumulated into ``runner.aug_grad_buffer``,
which PerturbationSensitivityAnalysisHookWithGradients consumes to build the
cross-correlation matrix.
"""

import os
import json
from copy import deepcopy

import torch

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS


@HOOKS.register_module()
class CollectGradientHook(Hook):
    """Probes ``d loss / d magnitude`` for each differentiable augmentation and
    accumulates the per-op scalar gradients into ``runner.aug_grad_buffer``.

    Args:
        probe_interval (int): Run a probe every this many training iters. Each
            probe costs one forward/backward per differentiable op, so keep this
            reasonably large. Defaults to 50.
        log_interval (int): Flush the latest gradients to disk every this many
            iters. Defaults to 200.
        ref_magnitude (float): Magnitude at which every op is probed (the
            "flank", not an extreme). Defaults to 0.5 (eps/2 for the diff
            module's default eps=1.0).
    """

    def __init__(
        self,
        probe_interval: int = 50,
        log_interval: int = 200,
        ref_magnitude: float = 0.5,
    ) -> None:
        self.probe_interval = probe_interval
        self.log_interval = log_interval
        self.ref_magnitude = ref_magnitude

        self.grad_buffer = None
        self.grad_log_path = None
        self._probe_loader = None
        self._probe_iter = None

    def before_run(self, runner: Runner) -> None:
        # Shared buffer, stashed on the runner so the correlation hook can read it.
        self.grad_buffer = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
        runner.aug_grad_buffer = self.grad_buffer

        self.grad_log_path = os.path.join(runner.cfg.work_dir, "aug_gradient_log.txt")
        if is_main_process():
            open(self.grad_log_path, "w+").close()  # clear if it already exists

        # Dedicated clean probe dataloader: the val config's pipeline has no
        # random augmentation and loads annotations, so it yields clean images
        # with GT labels. Independent of the val_loop dataloader (which
        # apply_perturbations_dataloader mutates in place during SA sweeps).
        dataloader_cfg = deepcopy(runner.cfg.val_dataloader)
        diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
        self._probe_loader = runner.build_dataloader(
            dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
        )
        self._probe_iter = iter(self._probe_loader)

    def _next_probe_batch(self):
        try:
            return next(self._probe_iter)
        except StopIteration:
            self._probe_iter = iter(self._probe_loader)
            return next(self._probe_iter)

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        # Temporal gate: only probe once the 1st SA loop has developed the
        # distribution. sa_curve is read purely as a boolean here.
        val_loop = getattr(runner, "val_loop", None)
        if getattr(val_loop, "sa_curve", None) is None:
            return

        if runner.iter % self.probe_interval != 0:
            return

        grads = self._probe_gradients(runner)
        if not grads:
            return

        for name, value in grads.items():
            self.grad_buffer[name].append(value)

        if runner.iter % self.log_interval == 0 and is_main_process():
            record = {"iter": int(runner.iter), "grads": grads}
            with open(self.grad_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _probe_gradients(self, runner: Runner) -> dict:
        """Returns {op_name: d loss / d magnitude} for all differentiable ops,
        measured on a single shared clean batch (so columns stay aligned)."""
        model = runner.model
        model = model.module if hasattr(model, "module") else model

        was_training = model.training
        # eval() freezes BatchNorm running stats (a train-mode forward would
        # mutate them) and, for SyncBN under DDP, avoids the cross-rank sync.
        model.eval()

        grads = {}
        try:
            data = self._next_probe_batch()
            # data_preprocessor casts to device, bgr->rgb, and normalizes.
            data = model.data_preprocessor(data, training=True)
            inputs = data["inputs"]
            data_samples = data["data_samples"]

            # De-normalize to [0, 1] RGB, the diff-aug ops' input contract.
            mean = model.data_preprocessor.mean.view(1, -1, 1, 1)
            std = model.data_preprocessor.std.view(1, -1, 1, 1)
            rgb01 = ((inputs * std + mean) / 255.0).clamp(0.0, 1.0)

            for name, op in DIFFERENTIABLE_PERTURBATIONS.items():
                grads[name] = self._grad_for_op(
                    model, op, rgb01, mean, std, data_samples
                )
        except Exception as e:  # noqa: BLE001
            print_log(
                f"CollectGradientHook probe failed at iter {runner.iter}: {e}",
                logger="current",
            )
            grads = {}
        finally:
            if was_training:
                model.train()

        return grads

    def _grad_for_op(self, model, op, rgb01, mean, std, data_samples) -> float:
        delta = torch.tensor(
            self.ref_magnitude,
            dtype=rgb01.dtype,
            device=rgb01.device,
            requires_grad=True,
        )
        perturbed01 = op(rgb01, delta)
        # Re-normalize back into the model's expected (normalized) input space.
        perturbed = (perturbed01 * 255.0 - mean) / std

        loss_dict = model.loss(perturbed, data_samples)
        loss = sum(
            _sum_loss_value(v) for k, v in loss_dict.items() if "loss" in k
        )

        grad = torch.autograd.grad(loss, delta)[0]
        return float(grad.detach().cpu().item())


def _sum_loss_value(value):
    """Mirror mmengine parse_losses: a loss entry is a Tensor or a list of
    Tensors; reduce to a scalar."""
    if torch.is_tensor(value):
        return value.mean()
    return sum(v.mean() for v in value)
