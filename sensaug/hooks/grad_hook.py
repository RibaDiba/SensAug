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

The magnitude is a length-B vector -- one delta PER IMAGE -- so a single probe
yields one sensitivity number per image per op, not one batch-averaged number per
op. That is what lets PerturbationSensitivityAnalysisHookWithGradients correlate
augmentations across IMAGES ("do two augs hurt the same images", i.e. redundancy)
rather than across training time ("do two augs get less sensitive at the same
moments", i.e. convergence drift, which is what a batch-averaged scalar can only
ever measure). Per-image independence rests on the probe running in eval() --
see _probe_gradients.

The per-op, per-image gradients are accumulated into ``runner.aug_grad_buffer``
as one (B,) array per probe; the buffer keeps probe boundaries intact (a list of
arrays, not one flat array) because the correlation hook's cluster bootstrap
resamples whole probes.
"""

import os
import json
from copy import deepcopy

import numpy as np
import torch

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS


class ProbeError(RuntimeError):
    """A gradient probe produced something structurally wrong (bad shape, NaN).

    Distinct from a transient failure (OOM, dataloader hiccup) so that
    _probe_gradients can let it propagate instead of degrading it into a
    silently skipped probe.
    """


@HOOKS.register_module()
class CollectGradientHook(Hook):
    """Probes ``d loss / d magnitude`` for each differentiable augmentation and
    accumulates the per-image gradients into ``runner.aug_grad_buffer``.

    Args:
        probe_interval (int): Run a probe every this many training iters. Each
            probe costs one forward/backward per differentiable op, so keep this
            reasonably large. Defaults to 50.
        log_interval (int): Flush accumulated probe records to disk every this
            many iters. This is a flush cadence, NOT a sampling filter -- every
            probe is logged. Defaults to 200.
        ref_magnitude (float): Magnitude at which every op is probed (the
            "flank", not an extreme). Defaults to 0.5 (eps/2 for the diff
            module's default eps=1.0). NOTE this number is not commensurable
            across ops: for the 12 photometric ops it means "halfway to the
            rail" (large), while for ``blur`` the magnitude IS sigma (so 0.5 is
            a mild blur) and for ``noise`` it is a std that clamps most pixels.
            Sweep it before trusting cross-op comparisons.
        per_image_delta (bool): If True (default), probe with a length-B delta
            and store one gradient per image. If False, use the old scalar delta
            and store one batch-averaged number per probe -- kept runnable for
            A/B comparison only; it cannot support a redundancy matrix.
        probe_seed (int): RNG seed applied inside each probe so that ``noise``
            draws the same sample every time. The RNG state is saved and restored
            around the probe, so the training stream is unaffected. Defaults to 0.
    """

    def __init__(
        self,
        probe_interval: int = 50,
        log_interval: int = 200,
        ref_magnitude: float = 0.5,
        per_image_delta: bool = True,
        probe_seed: int = 0,
    ) -> None:
        self.probe_interval = probe_interval
        self.log_interval = log_interval
        self.ref_magnitude = ref_magnitude
        self.per_image_delta = per_image_delta
        self.probe_seed = probe_seed

        self.grad_buffer = None
        self.grad_log_path = None
        self._probe_loader = None
        self._probe_iter = None
        self._pending_records = []

    def before_run(self, runner: Runner) -> None:
        # Shared buffer, stashed on the runner so the correlation hook can read
        # it. One entry per probe per op, each a (B,) array -- probe boundaries
        # are load-bearing for the cluster bootstrap, so do NOT flatten.
        self.grad_buffer = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
        runner.aug_grad_buffer = self.grad_buffer

        self.grad_log_path = os.path.join(runner.cfg.work_dir, "aug_gradient_log.txt")
        if is_main_process():
            # Append when resuming so a resumed run does not throw away the
            # probes the previous run already logged (corr_matrix_log.txt is
            # append-only, and these two need to stay consistent with each other).
            mode = "a" if getattr(runner, "_resume", False) else "w"
            open(self.grad_log_path, mode).close()

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

        if is_main_process():
            # Log EVERY probe, in full: this makes R recomputable offline from
            # the log without retraining. .tolist() is required -- np.ndarray and
            # np.float32 both raise TypeError in json.dumps.
            self._pending_records.append(
                {
                    "iter": int(runner.iter),
                    "batch_size": int(next(iter(grads.values())).shape[0]),
                    "grads": {name: value.tolist() for name, value in grads.items()},
                }
            )
            if runner.iter % self.log_interval == 0:
                self._flush_records()

    def after_run(self, runner: Runner) -> None:
        if is_main_process():
            self._flush_records()

    def _flush_records(self) -> None:
        if not self._pending_records:
            return
        with open(self.grad_log_path, "a") as f:
            for record in self._pending_records:
                f.write(json.dumps(record) + "\n")
        self._pending_records.clear()

    def _probe_gradients(self, runner: Runner) -> dict:
        """Returns {op_name: (B,) array of d loss / d magnitude, one per image}
        for all differentiable ops, measured on a single shared clean batch (so
        columns stay aligned across ops)."""
        model = runner.model
        model = model.module if hasattr(model, "module") else model

        was_training = model.training
        # eval() freezes BatchNorm running stats (a train-mode forward would
        # mutate them) and, for SyncBN under DDP, avoids the cross-rank sync.
        #
        # It is ALSO what makes the per-image gradients mean anything. With BN in
        # train mode, every image's output depends on the whole batch's
        # statistics, so d loss / d delta[i] would absorb the other images'
        # deltas and the "which images does this aug hurt" signal -- the entire
        # point of the per-image probe -- would be contaminated. In eval mode
        # image i's loss depends only on delta[i].
        #
        # Same reason the data preprocessor must not apply batch-level
        # augmentation (Mixup/CutMix would couple images). Stock
        # SegDataPreProcessor (pad + normalize) is fine.
        model.eval()
        assert not model.training, (
            "gradient probe must run in eval mode: train-mode BatchNorm couples "
            "images through the batch statistics and destroys per-image "
            "independence"
        )

        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        grads = {}
        try:
            # Fix the probe RNG so `noise` draws the SAME sample every probe.
            # Its gradient is a projection of the pixel gradient onto that
            # sample, so a fresh draw per probe would turn its column into
            # measurement noise and attenuate every correlation it appears in.
            # Restored in `finally`, so the training RNG stream is untouched.
            torch.manual_seed(self.probe_seed)

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
                    name, model, op, rgb01, mean, std, data_samples
                )
        except ProbeError:
            # A wrong-shaped or non-finite gradient is a bug in the probe, not a
            # transient failure. Fail loudly rather than degrading into a
            # silently skipped probe (which is what the blanket except below
            # would otherwise do).
            raise
        except Exception as e:  # noqa: BLE001
            print_log(
                f"CollectGradientHook probe failed at iter {runner.iter}: {e}",
                logger="current",
            )
            grads = {}
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if was_training:
                model.train()

        return grads

    def _grad_for_op(
        self, name, model, op, rgb01, mean, std, data_samples
    ) -> np.ndarray:
        """d loss / d magnitude for one op, as a (B,) array -- one sensitivity
        per image. (In the legacy scalar mode, a (1,) array holding the old
        batch-averaged number.)"""
        batch_size = rgb01.shape[0]
        if self.per_image_delta:
            delta = torch.full(
                (batch_size,),
                self.ref_magnitude,
                dtype=rgb01.dtype,
                device=rgb01.device,
                requires_grad=True,
            )
        else:
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
        loss = sum(_sum_loss_value(v) for k, v in loss_dict.items() if "loss" in k)

        # allow_unused defaults to False, so an op that drops delta from the graph
        # entirely already raises here rather than returning None.
        grad = torch.autograd.grad(loss, delta)[0]

        # Defensive only: autograd returns a gradient shaped like `delta`, so this
        # cannot fire today. It is a tripwire for a future change that reshapes
        # delta. It does NOT catch an op that collapses the per-image delta
        # internally (e.g. using delta.mean()) -- that yields a correctly shaped
        # but wrong gradient, and is caught instead by the per-image independence
        # test in tests/test_grad_hook.py.
        expected_shape = (batch_size,) if self.per_image_delta else ()
        if tuple(grad.shape) != expected_shape:
            raise ProbeError(
                f"op {name!r}: expected a gradient of shape {expected_shape}, got "
                f"{tuple(grad.shape)}"
            )
        if not torch.isfinite(grad).all():
            raise ProbeError(f"op {name!r}: non-finite gradient {grad.tolist()}")

        return grad.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _sum_loss_value(value):
    """Mirror mmengine parse_losses: a loss entry is a Tensor or a list of
    Tensors; reduce to a scalar.

    NOTE this batch-mean gives image i a weight proportional to its valid-pixel
    count (mmseg's CE with ignore_index=255 normalizes by the batch's TOTAL
    valid-pixel count), so the stored per-image gradients carry a shared
    per-image scale factor. That is not a constant, and Pearson is not invariant
    to it -- it is removed downstream by the scale normalization in
    grad_sens_analysis.calculate_cross_corelation, which handles any per-image
    multiplicative factor (including plain image difficulty) in one place."""
    if torch.is_tensor(value):
        return value.mean()
    return sum(v.mean() for v in value)
