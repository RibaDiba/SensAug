"""
Hook for probing and storing the gradient of the training loss with respect to
each differentiable augmentation's magnitude.

The differentiable augmentation ops (sensaug.dataset.differentiable_augmentations)
are autograd-compatible but are NOT wired into the training step -- there is no
persistent magnitude parameter whose ``.grad`` we could read after a real
``train_step`` (mmengine's OptimWrapper already ran ``zero_grad`` by the time
``after_train_iter`` fires). So this hook *produces* the gradient itself: it takes
a clean batch, applies each differentiable op at a fixed reference magnitude, runs
a forward pass, and reads ``d loss / d magnitude`` via ``torch.autograd.grad``
(which never populates model parameter ``.grad``, so training is untouched).

The measurement is a FROZEN FRAME. Every ``interval`` train iters the hook pauses
on one model state and sweeps the ENTIRE clean val set in a single pass, so every
column of the resulting (A ops x N images) matrix describes the same model. The
earlier design streamed one batch every ``probe_interval`` train iters and pooled
them into a window; that spread a window's columns across hundreds of model states
and revisited the same images at different states, so R mixed augmentation
redundancy with convergence drift. The sweep is also *cheaper*: the streaming probe
re-probed the val shard ~10x over a run, where four sweeps cover it four times.

The sweep runs from ``after_train_iter`` on its OWN iteration clock, deliberately
independent of the sensitivity-analysis pipeline in sensaug/loops.py. It used to
fire from ``after_val_epoch``, which under ``--aug-type=ours`` is called by
RobustValLoop itself -- so the correlation measurement was nested inside the SA
measurement, could only happen on an SA round, and did not exist at all for any
other aug type. They answer different questions and now keep different clocks; R
can be measured against an ``--aug-type=none`` baseline.

The magnitude is a length-B vector -- one delta PER IMAGE -- so a single batch
yields one sensitivity number per image per op, not one batch-averaged number per
op. That is what lets PerturbationSensitivityAnalysisHookWithGradients correlate
augmentations across IMAGES ("do two augs hurt the same images", i.e. redundancy)
rather than across training time. Per-image independence rests on the sweep
running in eval() -- see _sweep.

The per-op, per-image gradients are accumulated into ``runner.aug_grad_buffer``
as one (B,) array per sweep batch; the buffer keeps batch boundaries intact (a
list of arrays, not one flat array) because the correlation hook's cluster
bootstrap resamples whole batches -- images in a batch share the batch-mean loss's
valid-pixel scale factor, so they are not fully independent draws.
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


def iteration_count(runner) -> int:
    """The 1-based count of training iterations completed at an ``after_train_iter``.

    ``runner.iter`` is the 0-based index of the iteration that just finished --
    IterBasedTrainLoop increments ``_iter`` AFTER this hook point -- so the count is
    one more than it. (Same convention as mmengine's ``Hook.every_n_train_iters``.)
    Off by one here and every sweep lands one iteration away from where the config
    says it does, which nothing would ever catch.
    """
    return runner.iter + 1


def training_progress(runner) -> float:
    """Fraction of training completed, in [0, 1]. Logged as the ``checkpoint`` field
    of both this pipeline's logs, so the two stay labelled consistently."""
    max_iters = getattr(runner, "max_iters", 0) or 0
    return iteration_count(runner) / max_iters if max_iters else 0.0


def fires_at(runner, interval: int) -> bool:
    """Whether the correlation pipeline runs at this iteration.

    ONE definition, imported by both halves of the pipeline (the sweep in this
    module and the R emission in grad_sens_analysis), because they must fire on the
    same iteration: the analyser correlates the sweep the collector just took, and a
    gate that disagreed by one iteration would hand it an empty -- or a stale --
    buffer.

    The final iteration always fires, so the end-of-training R exists even when
    ``max_iters`` is not a multiple of ``interval``. The two conditions are ORed
    into one predicate rather than checked separately, so that final iteration fires
    exactly once in the common case where it satisfies both.

    The predicate is a pure function of ``runner.iter``, which is identical on every
    rank. That is what makes the analyser's ``all_gather_object`` safe: no rank can
    skip an emission that the others take, so the collective cannot deadlock.
    """
    iteration = iteration_count(runner)
    max_iters = getattr(runner, "max_iters", 0) or 0
    return iteration % interval == 0 or iteration == max_iters


class ProbeError(RuntimeError):
    """A gradient probe produced something structurally wrong (bad shape, NaN).

    Distinct from a transient failure (OOM, dataloader hiccup) so that the sweep
    can let it propagate instead of degrading it into a silently skipped batch.
    """


@HOOKS.register_module()
class CollectGradientHook(Hook):
    """Sweeps the whole clean val set on a frozen model every ``interval`` train
    iters, probing ``d loss / d magnitude`` for every differentiable augmentation,
    and accumulates the per-image gradients into ``runner.aug_grad_buffer``.

    Must run BEFORE PerturbationSensitivityAnalysisHookWithGradients, which
    consumes the buffer in its own ``after_train_iter`` and must see the sweep this
    hook just wrote. Both hooks must be given the SAME ``interval``; register this
    one at a higher priority (see train.py).

    Args:
        interval (int): Train iters between sweeps. Must match the correlation
            hook's ``interval``. Each sweep is one frozen model state, which is
            what makes R a statement about a single model.
        sweep_batch_size (int): Images per forward during the sweep. Larger is
            faster (better GPU utilization) but multiplies activation memory --
            the backward reaches the input, and val images are full resolution.
            This is also the cluster the downstream bootstrap resamples, so at 1
            the clusters are singletons (an i.i.d. image bootstrap). Defaults to 1.
        ref_magnitude (float): Magnitude at which every op is probed (the
            "flank", not an extreme). Defaults to 0.5 (eps/2 for the diff
            module's default eps=1.0). NOTE this number is not commensurable
            across ops: for the 12 photometric ops it means "halfway to the
            rail" (large), while for ``blur`` the magnitude IS sigma (so 0.5 is
            a mild blur) and for ``noise`` it is a std that clamps most pixels.
            Sweep it before trusting cross-op comparisons.
        per_image_delta (bool): If True (default), probe with a length-B delta
            and store one gradient per image. If False, use the old scalar delta
            and store one batch-averaged number per batch -- kept runnable for
            A/B comparison only; it cannot support a redundancy matrix.
        probe_seed (int): RNG seed applied before each batch so that ``noise``
            draws the same sample every time. The RNG state is saved and restored
            around the sweep, so the training stream is unaffected. Defaults to 0.
    """

    def __init__(
        self,
        interval: int,
        sweep_batch_size: int = 1,
        ref_magnitude: float = 0.5,
        per_image_delta: bool = True,
        probe_seed: int = 0,
    ) -> None:
        if interval < 1:
            raise ValueError(f"interval must be a positive iteration count, got {interval}")
        self.interval = interval
        self.sweep_batch_size = sweep_batch_size
        self.ref_magnitude = ref_magnitude
        self.per_image_delta = per_image_delta
        self.probe_seed = probe_seed

        self.grad_buffer = None
        self.grad_log_path = None
        self._probe_loader = None
        self._pending_records = []

    def before_run(self, runner: Runner) -> None:
        # Shared buffer, stashed on the runner so the correlation hook can read
        # it. One entry per sweep batch per op, each a (B,) array -- batch
        # boundaries are load-bearing for the cluster bootstrap, so do NOT flatten.
        self.grad_buffer = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
        runner.aug_grad_buffer = self.grad_buffer

        self.grad_log_path = os.path.join(runner.cfg.work_dir, "aug_gradient_log.txt")
        if is_main_process():
            # Append when resuming so a resumed run does not throw away the
            # sweeps the previous run already logged (corr_matrix_log.json is
            # append-only, and these two need to stay consistent with each other).
            mode = "a" if getattr(runner, "_resume", False) else "w"
            open(self.grad_log_path, mode).close()

        # Dedicated clean probe dataloader: the val config's pipeline has no
        # random augmentation and loads annotations, so it yields clean images
        # with GT labels. Independent of the val_loop dataloader (which
        # apply_perturbations_dataloader mutates in place during SA sweeps).
        dataloader_cfg = deepcopy(runner.cfg.val_dataloader)
        dataloader_cfg.batch_size = self.sweep_batch_size
        diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
        self._probe_loader = runner.build_dataloader(
            dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
        )

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if not fires_at(runner, self.interval):
            return

        # A safe point to probe: train_step already ran backward + step + zero_grad
        # through the OptimWrapper, so there is no pending gradient for the probe to
        # clobber -- and torch.autograd.grad populates no parameter .grad anyway.
        self._sweep(runner, training_progress(runner))

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

    def _sweep(self, runner: Runner, checkpoint: float) -> None:
        """One frozen-frame pass over the entire probe dataloader.

        The model is not stepped anywhere in here -- ``torch.autograd.grad``
        computes d loss / d delta without populating parameter ``.grad`` -- so
        every image is scored against the SAME weights. That is the whole point:
        a column of D_grad is then attributable to the image, not to whenever
        during training it happened to be drawn.
        """
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
            "gradient sweep must run in eval mode: train-mode BatchNorm couples "
            "images through the batch statistics and destroys per-image "
            "independence"
        )

        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        n_batches = n_images = 0
        try:
            for data in self._probe_loader:
                try:
                    grads = self._probe_batch(model, data)
                except ProbeError:
                    # A wrong-shaped or non-finite gradient is a bug in the probe,
                    # not a transient failure. Fail loudly rather than degrading
                    # into a silently skipped batch (which is what the blanket
                    # except below would otherwise do).
                    raise
                except Exception as e:  # noqa: BLE001
                    print_log(
                        f"CollectGradientHook batch failed during the "
                        f"{checkpoint:.0%} sweep: {e}",
                        logger="current",
                    )
                    continue

                for name, value in grads.items():
                    self.grad_buffer[name].append(value)

                batch_size = int(next(iter(grads.values())).shape[0])
                n_batches += 1
                n_images += batch_size

                if is_main_process():
                    # Log EVERY batch, in full: this makes R recomputable offline
                    # from the log without retraining. .tolist() is required --
                    # np.ndarray and np.float32 both raise TypeError in json.dumps.
                    self._pending_records.append(
                        {
                            "checkpoint": checkpoint,
                            "iter": int(runner.iter),
                            "batch_size": batch_size,
                            "grads": {
                                name: value.tolist() for name, value in grads.items()
                            },
                        }
                    )
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if was_training:
                model.train()

        if is_main_process():
            self._flush_records()

        print_log(
            f"[grad-sweep] checkpoint {checkpoint:.0%} (iter {runner.iter}): swept "
            f"{n_images} images in {n_batches} batches on this rank",
            logger="current",
        )

    def _probe_batch(self, model, data) -> dict:
        """Returns {op_name: (B,) array of d loss / d magnitude, one per image}
        for all differentiable ops, measured on a single shared clean batch (so
        columns stay aligned across ops).

        Assumes the caller has already put the model in eval mode and taken
        responsibility for restoring the RNG (see _sweep).
        """
        # Fix the probe RNG so `noise` draws the SAME sample for every batch. Its
        # gradient is a projection of the pixel gradient onto that sample, so a
        # fresh draw per batch would turn its row into measurement noise and
        # attenuate every correlation it appears in.
        torch.manual_seed(self.probe_seed)

        # data_preprocessor casts to device, bgr->rgb, and normalizes. It pads but
        # never crops (stack_batch uses max(size - shape, 0)), so val images stay
        # at their full resolution here.
        data = model.data_preprocessor(data, training=True)
        inputs = data["inputs"]
        data_samples = data["data_samples"]

        # De-normalize to [0, 1] RGB, the diff-aug ops' input contract.
        mean = model.data_preprocessor.mean.view(1, -1, 1, 1)
        std = model.data_preprocessor.std.view(1, -1, 1, 1)
        rgb01 = ((inputs * std + mean) / 255.0).clamp(0.0, 1.0)

        return {
            name: self._grad_for_op(name, model, op, rgb01, mean, std, data_samples)
            for name, op in DIFFERENTIABLE_PERTURBATIONS.items()
        }

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
