"""
Hook for probing and storing the gradient of the training loss with respect to
each differentiable augmentation's magnitude.

The differentiable augmentation ops (sensaug.dataset.differentiable_augmentations)
are autograd-compatible but are NOT wired into the training step -- there is no
persistent magnitude parameter whose ``.grad`` we could read after a real
``train_step`` (mmengine's OptimWrapper already ran ``zero_grad`` by the time
``after_train_iter`` fires). So this hook *produces* the gradient itself: it takes
a clean batch, applies each differentiable op at a chosen magnitude, runs a forward
pass, and reads ``d loss / d magnitude`` via ``torch.autograd.grad`` (which never
populates model parameter ``.grad``, so training is untouched).

WHERE THE MAGNITUDE COMES FROM. It used to be one hardcoded constant (0.5) for
every op, which is not commensurable across them: at 0.5, ``blur`` moves a pixel by
at most 78/255 while ``noise`` moves it by 251/255, so R correlated ops held at
arbitrary relative strengths. The magnitude is now taken from the per-op
distribution the SA loop publishes to ``runner.corr_magnitudes`` (see
sensaug/corr_magnitudes.py for the modes, sensaug/loops.py for the publisher).
The snapshot is read ONCE per sweep, so one R is always one magnitude regime, and
falls back to ``ref_magnitude`` whenever no snapshot exists -- before the first
post-warmup SA round, and for the whole run when the SA loop is disabled. Every
emission is labelled with which of the two it was, because matrices measured under
the two are not comparable.

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
# The merged 32-op vocabulary (the base module's 14 + the 18 AutoAugment-family
# ops), aliased to the old name so every sweep site below is unchanged. See
# ALL_DIFFERENTIABLE_PERTURBATIONS for the label caveat on the geometric ops.
from sensaug.dataset.differentiable_augmentations_aa import (
    ALL_DIFFERENTIABLE_PERTURBATIONS as DIFFERENTIABLE_PERTURBATIONS,
)
from sensaug.corr_magnitudes import (
    MAGNITUDE_MODES,
    MODE_FIXED,
    MODE_MODAL,
    resolve_magnitudes,
)


def iteration_count(runner) -> int:
    """
    Determine the one-based number of completed training iterations.
    
    Returns:
    	int: The completed iteration count.
    """
    return runner.iter + 1


def training_progress(runner) -> float:
    """
    Measures the fraction of training iterations completed.
    
    Returns:
    	float: The completed-training fraction, or 0.0 when the maximum iteration count is unavailable.
    """
    max_iters = getattr(runner, "max_iters", 0) or 0
    return iteration_count(runner) / max_iters if max_iters else 0.0


def fires_at(runner, interval: int) -> bool:
    """Determine whether the current iteration is scheduled for processing.
    
    Parameters:
        interval (int): Number of iterations between scheduled runs.
    
    Returns:
        bool: ``True`` on interval multiples or the final iteration, ``False`` otherwise.
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
        ref_magnitude (float): FALLBACK magnitude, used for any op the SA
            snapshot does not cover and for every op when there is no snapshot
            at all (``magnitude_mode="fixed"``, ``--no-corr-sa``, or the rounds
            before the first SA completes). Defaults to 0.5 (eps/2 for the diff
            module's default eps=1.0).

            NOTE this number is not commensurable across ops: for the 12
            photometric ops it means "halfway to the rail" (large), while for
            ``blur`` the magnitude IS sigma (so 0.5 is a mild blur) and for
            ``noise`` it is a std that saturates most pixels. Measured on a
            random image, magnitude 0.5 moves a pixel by at most 78/255 under
            ``blur`` and 251/255 under ``noise``. That incommensurability is
            precisely what ``magnitude_mode`` exists to remove -- an R built on
            this fallback correlates ops held at arbitrary relative strengths.
        magnitude_mode (str): Where each op's probe magnitude comes from.

            - ``"mode"`` (default): the modal level of that op's SA distribution,
              constant across the batch. R stays a statement about image
              variation alone.
            - ``"sampled_shared"``: drawn per image from the op's distribution
              the way training draws it, but with one shared quantile and jitter
              deviate per image across all ops -- common random numbers, so the
              sampling does not decorrelate the rows.
            - ``"sampled_independent"``: every op draws on its own. Most faithful
              to training, but injects row-uncorrelated noise that attenuates
              every cell of R toward zero by an unknown, pdf-dependent factor,
              so R stops being comparable across checkpoints.
            - ``"fixed"``: always ``ref_magnitude``; reproduces the pre-snapshot
              behaviour exactly.

            Every mode falls back to ``ref_magnitude`` when no snapshot exists.
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
        magnitude_mode: str = MODE_MODAL,
        magnitudes_path: str = None,
    ) -> None:
        """
        Configure periodic gradient probing and augmentation-magnitude sampling.
        
        Parameters:
            interval (int): Number of training iterations between gradient sweeps.
            sweep_batch_size (int): Number of validation images processed per probe batch.
            ref_magnitude (float): Fallback augmentation magnitude when no sampled value is available.
            per_image_delta (bool): Whether to compute a separate magnitude gradient for each image.
            probe_seed (int): Seed used for reproducible magnitude sampling during sweeps.
            magnitude_mode (str): Strategy used to obtain augmentation magnitudes.
            magnitudes_path (str, optional): Path to a JSON snapshot providing seeded magnitudes.
        
        Raises:
            ValueError: If `interval` is less than one or `magnitude_mode` is unsupported.
        """
        if interval < 1:
            raise ValueError(f"interval must be a positive iteration count, got {interval}")
        if magnitude_mode not in MAGNITUDE_MODES:
            raise ValueError(
                f"unknown magnitude_mode {magnitude_mode!r}, expected one of "
                f"{MAGNITUDE_MODES}"
            )
        self.interval = interval
        self.sweep_batch_size = sweep_batch_size
        self.ref_magnitude = ref_magnitude
        self.per_image_delta = per_image_delta
        self.probe_seed = probe_seed
        self.magnitude_mode = magnitude_mode
        self.magnitudes_path = magnitudes_path
        # Seeded from magnitudes_path in before_run; used only when this run's own
        # SA loop has not published anything (SA disabled, or pre-first-round).
        self._seed_snapshot = {}

        self.grad_buffer = None
        self.grad_log_path = None
        self._probe_loader = None
        self._pending_records = []
        # Dedicated Generator rather than the global numpy RNG: magnitude draws
        # must not consume from (and so perturb) the training data stream. Reseeded
        # at the start of each sweep so a sweep is reproducible, then allowed to
        # advance ACROSS batches -- reseeding per batch would hand every batch the
        # same draw, which at sweep_batch_size=1 collapses the sampled modes to a
        # single magnitude for the entire val set.
        self._magnitude_rng = None

    def _load_seed_snapshot(self) -> dict:
        """
        Load the latest augmentation-magnitude snapshot from the configured file.
        
        Returns:
        	dict: The latest per-operation magnitude levels and probabilities, or an
        	empty dictionary when no snapshot file is configured.
        
        Raises:
        	ValueError: If the snapshot file contains no records or its latest record
        	has no magnitude data.
        """
        if not self.magnitudes_path:
            return {}

        with open(self.magnitudes_path) as f:
            records = json.load(f)
        if not records:
            raise ValueError(f"{self.magnitudes_path} holds no snapshots")

        magnitudes = records[-1].get("magnitudes")
        if not magnitudes:
            raise ValueError(
                f"{self.magnitudes_path}: last record has no 'magnitudes' key"
            )

        # Drop the recorded modal value; it is a derived summary, and keeping it
        # would shadow whatever magnitude_mode this run asks for.
        snapshot = {
            op: {"levels": entry["levels"], "probs": entry["probs"]}
            for op, entry in magnitudes.items()
        }
        print_log(
            f"[grad-sweep] seeded probe magnitudes for {len(snapshot)} ops from "
            f"{self.magnitudes_path} (iter {records[-1].get('iter')})",
            logger="current",
        )
        return snapshot

    def before_run(self, runner: Runner) -> None:
        # Shared buffer, stashed on the runner so the correlation hook can read
        # it. One entry per sweep batch per op, each a (B,) array -- batch
        # boundaries are load-bearing for the cluster bootstrap, so do NOT flatten.
        """
        Initialize gradient storage, logging, and the dedicated clean validation dataloader used for probing.
        """
        self.grad_buffer = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
        runner.aug_grad_buffer = self.grad_buffer

        self._seed_snapshot = self._load_seed_snapshot()

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
        """Flushes pending gradient records after the training run on the main process."""
        if is_main_process():
            self._flush_records()

    def _flush_records(self) -> None:
        """Append pending gradient records to the configured log file."""
        if not self._pending_records:
            return
        with open(self.grad_log_path, "a") as f:
            for record in self._pending_records:
                f.write(json.dumps(record) + "\n")
        self._pending_records.clear()

    def _sweep(self, runner: Runner, checkpoint: float) -> None:
        """
        Collect per-image augmentation gradients across the probe dataloader.
        
        The sweep evaluates all batches with a consistent magnitude snapshot and
        preserves the model and random-number-generator state afterward. Results and
        magnitude metadata are stored for downstream analysis.
        
        Parameters:
            runner (Runner): Training runner containing the model and probe state.
            checkpoint (float): Training progress associated with the sweep.
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

        # Whatever the SA loop published most recently -- read here, once, so every
        # batch in this sweep is probed at the SAME magnitudes. Reading it per batch
        # would let an SA round land mid-sweep and split one R across two magnitude
        # regimes. Empty until the first post-warmup SA round, and always empty when
        # the SA loop is disabled; both cases fall back to ref_magnitude.
        live = getattr(runner, "corr_magnitudes", None) or {}
        # A live SA snapshot wins over a seeded one: the seed exists to pin a run
        # that has no SA of its own, not to override one that does.
        snapshot = live or self._seed_snapshot
        self._magnitude_rng = np.random.default_rng(self.probe_seed)

        if self.magnitude_mode == MODE_FIXED or not snapshot:
            source = "fixed"
        elif live:
            source = "sa_snapshot"
        else:
            source = "seeded_snapshot"

        n_batches = n_images = 0
        swept_magnitudes = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
        try:
            for data in self._probe_loader:
                try:
                    grads, magnitudes = self._probe_batch(model, data, snapshot)
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
                for name, value in magnitudes.items():
                    swept_magnitudes[name].append(value)

                batch_size = int(next(iter(grads.values())).shape[0])
                n_batches += 1
                n_images += batch_size

                if is_main_process():
                    # Log EVERY batch, in full: this makes R recomputable offline
                    # from the log without retraining. .tolist() is required --
                    # np.ndarray and np.float32 both raise TypeError in json.dumps.
                    #
                    # The magnitudes go in alongside the gradients because a
                    # gradient is only interpretable together with the magnitude it
                    # was taken at -- without them an offline recompute cannot tell
                    # a fixed-0.5 sweep from an SA-driven one.
                    self._pending_records.append(
                        {
                            "checkpoint": checkpoint,
                            "iter": int(runner.iter),
                            "batch_size": batch_size,
                            "magnitude_source": source,
                            "magnitude_mode": self.magnitude_mode,
                            "grads": {
                                name: value.tolist() for name, value in grads.items()
                            },
                            "magnitudes": {
                                name: value.tolist()
                                for name, value in magnitudes.items()
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

        # Handed to PerturbationSensitivityAnalysisHookWithGradients, which records
        # it with R. Without it a fixed-0.5 matrix and an SA-driven matrix are
        # indistinguishable in the log, and the two are NOT comparable -- the whole
        # reason for this pipeline is that 0.5 means something different per op.
        runner.aug_grad_magnitude_info = {
            "source": source,
            "mode": self.magnitude_mode,
            "ref_magnitude": self.ref_magnitude,
            "per_op": {
                name: _magnitude_summary(values)
                for name, values in swept_magnitudes.items()
                if values
            },
        }

        print_log(
            f"[grad-sweep] checkpoint {checkpoint:.0%} (iter {runner.iter}): swept "
            f"{n_images} images in {n_batches} batches on this rank "
            f"(magnitudes: {source}/{self.magnitude_mode})",
            logger="current",
        )

    def _probe_batch(self, model, data, snapshot):
        """
        Compute per-image loss gradients and augmentation magnitudes for a clean batch.
        
        Parameters:
        	data: A clean batch to preprocess and probe.
        	snapshot: Per-operation magnitude distributions, or an empty mapping to use the reference magnitude.
        
        Returns:
        	A tuple containing per-operation gradient arrays and magnitude arrays, each indexed by image in the batch.
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

        batch_size = int(rgb01.shape[0])
        if self.per_image_delta:
            magnitudes = resolve_magnitudes(
                snapshot,
                list(DIFFERENTIABLE_PERTURBATIONS),
                batch_size,
                self.magnitude_mode,
                self.ref_magnitude,
                rng=self._magnitude_rng,
            )
        else:
            # The legacy scalar path has one delta for the whole batch, so it
            # cannot carry a per-image magnitude. Kept on the fixed reference.
            magnitudes = {
                name: np.full(1, self.ref_magnitude, dtype=np.float64)
                for name in DIFFERENTIABLE_PERTURBATIONS
            }

        grads = {
            name: self._grad_for_op(
                name, model, op, rgb01, mean, std, data_samples, magnitudes[name]
            )
            for name, op in DIFFERENTIABLE_PERTURBATIONS.items()
        }
        return grads, magnitudes

    def _grad_for_op(
        self, name, model, op, rgb01, mean, std, data_samples, magnitude
    ) -> np.ndarray:
        """
        Compute each image's loss sensitivity to an augmentation magnitude.
        
        Parameters:
        	name (str): Name of the augmentation operation.
        	magnitude (array-like): Probe magnitude for each image in the batch.
        
        Returns:
        	np.ndarray: Per-image gradients, or a single batch-averaged gradient in legacy scalar mode.
        
        Raises:
        	ProbeError: If the gradient has an unexpected shape or contains non-finite values.
        """
        batch_size = rgb01.shape[0]
        if self.per_image_delta:
            delta = torch.tensor(
                magnitude,
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


def _magnitude_summary(values) -> dict:
    """
    Summarize per-image probe magnitudes for an augmentation operation.
    
    Parameters:
    	values: Magnitude arrays collected across the sweep.
    
    Returns:
    	dict: A dictionary containing the mean, minimum, and maximum magnitude.
    """
    flat = np.concatenate(values)
    return {
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def _sum_loss_value(value):
    """
    Reduce a loss value or collection of loss values to a scalar mean.
    
    Parameters:
        value: A tensor or an iterable of tensors representing loss values.
    
    Returns:
        The mean of the tensor, or the sum of the means for an iterable of tensors.
    """
    if torch.is_tensor(value):
        return value.mean()
    return sum(v.mean() for v in value)
