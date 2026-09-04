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
sensaug/corr_magnitudes.py for the modes, sensaug/loops/sensaug_loop.py for the
publisher).
The snapshot is read ONCE per sweep, so one R is always one magnitude regime, and
falls back to ``ref_magnitude`` whenever no snapshot exists -- before the first
post-warmup SA round, and for the whole run when the SA loop is disabled. Every
emission is labelled with which of the two it was, because matrices measured under
the two are not comparable.

The measurement is a FROZEN FRAME. At each firing iteration the hook pauses on one
model state and sweeps the ENTIRE clean val set in a single pass, so every column
of the resulting (A ops x N images) matrix describes the same model. The earlier
design streamed one batch every ``probe_interval`` train iters and pooled them into
a window; that spread a window's columns across hundreds of model states and
revisited the same images at different states, so R mixed augmentation redundancy
with convergence drift. The sweep is also *cheaper*: the streaming probe re-probed
the val shard ~10x over a run, where a handful of sweeps cover it that many times.

WHICH iterations fire is decided in sensaug/round_schedule.py (a pure module, no
torch, so the arithmetic is testable on its own): by default the SA round grid, so that each sweep lands one hook point before an SA round that
can act on the resulting R. That is an alignment of schedules, not a coupling of
code -- the sweep still runs from ``after_train_iter`` and calls nothing in
sensaug/loops/. It used to fire from ``after_val_epoch``, which under
``--aug-type=ours`` is called by RobustValLoop itself, so the correlation
measurement was nested INSIDE the SA measurement and did not exist at all for any
other aug type. It still does not depend on a val loop existing: with
``--no-corr-sa`` the same iterations fire against a run that trains unaugmented,
which is exactly what makes the two arms comparable.

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
import contextlib
from copy import deepcopy

import numpy as np
import torch

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
# The merged 32-op vocabulary: the base module's 14 plus the 18 AutoAugment-family
# ops. `geometric_affine_matrix` / `warp_image_and_label` are the same pair
# sensaug.dataset.gpu_augment uses to keep image and label together -- shared on
# purpose, so the probe and the training pipeline cannot warp by different amounts.
from sensaug.dataset.differentiable_augmentations_aa import (
    DIFF32_OPS,
    geometric_affine_matrix,
    warp_image_and_label,
)
from sensaug.corr_magnitudes import (
    MAGNITUDE_MODES,
    MODE_FIXED,
    MODE_MODAL,
    resolve_magnitudes,
)


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


def fires_at(runner, interval: int = None, fire_iters=None) -> bool:
    """Whether the correlation pipeline runs at this iteration.

    ONE definition, imported by both halves of the pipeline (the sweep in this
    module and the R emission in grad_sens_analysis), because they must fire on the
    same iteration: the analyser correlates the sweep the collector just took, and a
    gate that disagreed by one iteration would hand it an empty -- or a stale --
    buffer.

    Two ways to say when, and exactly one of them is in play (see ``resolve_gate``):

    - ``fire_iters``: an explicit set of iteration counts. This is the DEFAULT for a
      training run, built by ``sensaug.round_schedule.resolve_schedule`` so that
      every emission lands one hook point before an SA round that can act on it.
      Nothing
      is added to the list -- in particular the final iteration does not fire,
      because the round it belongs to is the last one and no pdf follows it.
    - ``interval``: the plain modulo clock, kept for ``--corr-interval`` and for the
      offline recompute in scripts/compute_grad_corr.py. Here the final iteration
      ALWAYS fires, so the end-of-training R exists even when ``max_iters`` is not a
      multiple of ``interval``. The two conditions are ORed into one predicate
      rather than checked separately, so that final iteration fires exactly once in
      the common case where it satisfies both.

    Either way the predicate is a pure function of ``runner.iter``, which is
    identical on every rank. That is what makes the analyser's
    ``all_gather_object`` safe: no rank can skip an emission that the others take,
    so the collective cannot deadlock.
    """
    iteration = iteration_count(runner)
    if fire_iters is not None:
        return iteration in fire_iters
    max_iters = getattr(runner, "max_iters", 0) or 0
    return iteration % interval == 0 or iteration == max_iters


def resolve_gate(interval, fire_iters):
    """Normalize the two ways of specifying the clock into one ``(interval, iters)``.

    Exactly one is non-None on the way out, so both hooks store the same pair and
    hand it to the same ``fires_at`` -- there is no third code path where they could
    disagree about which mode they are in.

    An empty ``fire_iters`` raises rather than defaulting to "never fires": a
    correlation pipeline that is registered and can never emit is a configuration
    error, and it would otherwise surface as an experiment with an empty
    corr_matrix_log.json weeks later.
    """
    if interval is not None and fire_iters is not None:
        raise ValueError(
            f"interval={interval} and fire_iters={tuple(fire_iters)!r} were both "
            f"given; they are two different clocks and only one can be in effect"
        )
    if fire_iters is not None:
        iters = tuple(sorted({int(i) for i in fire_iters}))
        if not iters:
            raise ValueError(
                "fire_iters is empty, so the correlation pipeline would never emit"
            )
        if iters[0] < 1:
            raise ValueError(
                f"fire_iters must be positive iteration counts, got {iters[0]}"
            )
        return None, iters
    if interval is None or interval < 1:
        raise ValueError(f"interval must be a positive iteration count, got {interval}")
    return int(interval), None


class ProbeError(RuntimeError):
    """A gradient probe produced something structurally wrong (bad shape, NaN).

    Distinct from a transient failure (OOM, dataloader hiccup) so that the sweep
    can let it propagate instead of degrading it into a silently skipped batch.
    """


def _gather_gt(data_samples, op_name: str) -> torch.Tensor:
    """(B, 1, H, W) ground-truth label batch for a geometric op's label warp.

    Raises rather than falling back to an unwarped label: an image warped without
    its label yields a loss dominated by misalignment, and the resulting R column
    looks like a real measurement. That silent version of this bug is exactly what
    the warp exists to remove.
    """
    labels = []
    for sample in data_samples:
        gt = getattr(sample, "gt_sem_seg", None)
        if gt is None:
            raise ProbeError(
                f"op {op_name!r} is geometric and needs to warp the label alongside "
                "the image, but this batch carries no gt_sem_seg."
            )
        labels.append(gt.data)
    shapes = {tuple(x.shape) for x in labels}
    if len(shapes) != 1:
        raise ProbeError(
            f"op {op_name!r} needs a stackable label batch to warp; got mixed "
            f"shapes {sorted(shapes)}."
        )
    return torch.stack(labels, dim=0)


@contextlib.contextmanager
def _swapped_gt(data_samples, labels):
    """Temporarily install `labels` as the batch's ground truth.

    Restored on the way out because `_probe_batch` reuses one `data_samples` list
    across all 32 ops -- a geometric op that left its warped label behind would
    corrupt every op measured after it.
    """
    if labels is None:
        yield
        return
    originals = [sample.gt_sem_seg.data for sample in data_samples]
    try:
        for sample, label in zip(data_samples, labels):
            sample.gt_sem_seg.data = label
        yield
    finally:
        for sample, original in zip(data_samples, originals):
            sample.gt_sem_seg.data = original


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
        interval (int): Train iters between sweeps, when the pipeline is on the
            plain modulo clock (``--corr-interval``, or the offline recompute).
            Must match the correlation hook's ``interval``. Each sweep is one
            frozen model state, which is what makes R a statement about a single
            model. Mutually exclusive with ``fire_iters``.
        fire_iters (Sequence[int]): The explicit iteration counts to sweep on --
            what a training run uses by default, built from the SA round grid by
            ``sensaug.round_schedule.resolve_schedule`` so each sweep lands one hook
            point before an SA round that can act on it. Must match the
            correlation hook's ``fire_iters``.
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
        skip_pruned (bool): If True, skip the forward+backward for any op
            currently listed in ``runner.corr_pruned_ops`` (published by
            ``GradCorrValLoop`` under a ``HARD_PRUNING_METHODS`` arm, e.g. mRMR)
            EXCEPT on a ``full_recheck_iters`` iteration, when every op is
            measured regardless. A skipped op's row is backfilled from its last
            real measurement (never fabricated, never NaN -- see
            ``_last_full_row``), re-chunked to this sweep's real batch sizes, so
            R still has a same-length, correctly-clustered row for it and it can
            still compete to un-prune next round. Defaults to False: today's
            behaviour, every op measured every sweep.
        full_recheck_iters (Sequence[int]): The subset of ``fire_iters`` on
            which ``skip_pruned`` is ignored and every op is measured regardless
            -- normally the iterations that coincide with an SA-curve recompute
            round (``sensaug.round_schedule.resolve_schedule``'s
            ``full_recheck_iters``). A STATIC set fixed at config-build time
            rather than a live runner flag: which of this hook's firing
            iterations are recompute rounds is schedule arithmetic, knowable
            before training starts, and reading it as a runtime signal instead
            would lag by one round relative to that same iteration's own val
            loop (``after_train_iter`` hooks run before ``val_loop.run()`` on a
            shared iteration -- see ``RobustIterBasedTrainLoop.run``). Only
            meaningful alongside an explicit ``fire_iters`` schedule -- empty
            (the default) on the plain interval clock (``--corr-sync-sa`` /
            ``--corr-interval``), where ``skip_pruned`` never gets a recheck
            iteration to land on.
    """

    def __init__(
        self,
        interval: int = None,
        sweep_batch_size: int = 1,
        ref_magnitude: float = 0.5,
        per_image_delta: bool = True,
        probe_seed: int = 0,
        magnitude_mode: str = MODE_MODAL,
        magnitudes_path: str = None,
        fire_iters=None,
        skip_pruned: bool = False,
        full_recheck_iters=(),
    ) -> None:
        if magnitude_mode not in MAGNITUDE_MODES:
            raise ValueError(
                f"unknown magnitude_mode {magnitude_mode!r}, expected one of "
                f"{MAGNITUDE_MODES}"
            )
        self.interval, self.fire_iters = resolve_gate(interval, fire_iters)
        self.sweep_batch_size = sweep_batch_size
        self.ref_magnitude = ref_magnitude
        self.per_image_delta = per_image_delta
        self.probe_seed = probe_seed
        self.magnitude_mode = magnitude_mode
        self.magnitudes_path = magnitudes_path
        self.skip_pruned = skip_pruned

        # Same validation posture as PerturbationSensitivityAnalysisHookWithGradients's
        # control_iters: a full-recheck iteration only means anything against an
        # explicit fire_iters schedule, and one outside that schedule could never
        # fire in the first place.
        full_recheck = frozenset(int(i) for i in full_recheck_iters or ())
        if full_recheck and self.fire_iters is None:
            raise ValueError(
                f"full_recheck_iters {sorted(full_recheck)} were given on the "
                f"plain interval clock. A full recheck is defined by which ROUND "
                f"it falls in, so it only means anything against an explicit "
                f"fire_iters schedule."
            )
        if self.fire_iters is not None and not full_recheck <= set(self.fire_iters):
            raise ValueError(
                f"full_recheck_iters {sorted(full_recheck - set(self.fire_iters))} "
                f"are not in fire_iters: a recheck that never fires can never "
                f"un-stale a pruned op."
            )
        self.full_recheck_iters = full_recheck
        # Seeded from magnitudes_path in before_run; used only when this run's own
        # SA loop has not published anything (SA disabled, or pre-first-round).
        self._seed_snapshot = {}

        # The last real (non-skipped) measurement per op -- what a skip round
        # backfills from. Populated the first time an op is ever measured (round
        # 3's mandatory control probe, at the latest, since pruning can't start
        # before round 4), never cleared. Magnitudes need no equivalent cache --
        # they're computed fresh for every op every sweep regardless of
        # skip_ops; see _probe_batch's call site.
        self._last_full_row = {}

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
        """Read the LAST snapshot out of a corr_magnitudes.json from a prior run.

        The last one, not the first: it is the most converged model's magnitudes,
        and the point of this file is to pin a control arm to the same magnitudes
        the comparison run used.

        Fails loudly on a missing or malformed file rather than degrading to the
        fixed fallback. Silently probing at 0.5 when you asked for a pinned
        magnitude produces a control that is not comparable to anything, and
        nothing downstream would reveal it.
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
        self.grad_buffer = {name: [] for name in DIFF32_OPS}
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
        if not fires_at(runner, self.interval, self.fire_iters):
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

        # Which ops to skip this sweep, and the fallback that keeps the skip
        # list from ever leaving an op with no measurement to backfill from: an
        # op named as pruned but never yet measured for real (shouldn't happen
        # once round 3's mandatory control probe has run, but a config could
        # still hit it, e.g. --no-warmup with a hand-pinned corr_rounds) is
        # simply measured this sweep instead of skipped. full_recheck_iters is
        # static (fixed at construction, see the class docstring for why), so
        # there is no runner state to read here at all.
        full_recheck = iteration_count(runner) in self.full_recheck_iters
        raw_skip_ops = (
            frozenset()
            if not self.skip_pruned or full_recheck
            else getattr(runner, "corr_pruned_ops", frozenset())
        )
        skip_ops = frozenset(op for op in raw_skip_ops if op in self._last_full_row)
        active_ops = {
            name: op for name, op in DIFF32_OPS.items() if name not in skip_ops
        }

        n_batches = n_images = 0
        # One entry per successfully-processed batch, in order -- what a
        # skipped op's backfill re-chunks its cached row into below, so its
        # buffer has the same chunk COUNT and sizes as every freshly-measured
        # op's this sweep.
        batch_sizes = []
        swept_magnitudes = {name: [] for name in DIFF32_OPS}
        try:
            for data in self._probe_loader:
                try:
                    grads, magnitudes = self._probe_batch(
                        model, data, snapshot, active_ops
                    )
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
                batch_sizes.append(batch_size)

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

        # Cache every freshly-measured op's row, then backfill anything skipped
        # from its last real one -- re-chunked into THIS sweep's real batch
        # sizes, not appended as one giant chunk. `stack_probe_buffer` derives
        # `probe_ids` (the bootstrap's cluster id) from `names[0]`'s chunk
        # boundaries alone; if the stale op happened to be first in DIFF32_OPS's
        # iteration order, a single-chunk backfill would collapse the whole
        # window to one cluster and degenerate the bootstrap for every cell of
        # R, not just this op's row. Re-chunking keeps every op's chunk count
        # identical every sweep regardless of which op (if any) is stale.
        #
        # Never NaN, never fabricated: a skipped op's row is real evidence (just
        # stale), which is what keeps it eligible to be ranked back in next
        # round rather than reading as "dropped, exempt" (see the class
        # docstring's skip_pruned entry).
        stale_ops = []
        for name in DIFF32_OPS:
            if name in active_ops:
                if self.grad_buffer[name]:
                    self._last_full_row[name] = np.concatenate(self.grad_buffer[name])
            else:
                cached = self._last_full_row[name]
                if cached.shape[0] != n_images:
                    # The probe dataloader is built once in before_run and
                    # iterated in the same order every sweep, so this shouldn't
                    # happen -- but degrade to a length-matched resize with a
                    # loud warning rather than trip stack_probe_buffer's assert.
                    print_log(
                        f"[grad-sweep] cached row for {name!r} is "
                        f"{cached.shape[0]} images, this sweep swept {n_images}; "
                        f"resizing the backfill to match",
                        logger="current",
                        level=30,
                    )
                    cached = np.resize(cached, n_images)
                chunks, offset = [], 0
                for size in batch_sizes:
                    chunks.append(cached[offset : offset + size])
                    offset += size
                self.grad_buffer[name] = chunks
                stale_ops.append(name)
        # Read by PerturbationSensitivityAnalysisHookWithGradients._emit so
        # corr_matrix_log.json records which ops in this emission were carried
        # forward rather than freshly measured -- same reasoning as
        # aug_grad_magnitude_info below: silently blending stale rows in with
        # fresh ones would make a matrix look more current than it is.
        runner.aug_grad_stale_ops = frozenset(stale_ops)

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
            f"(magnitudes: {source}/{self.magnitude_mode})"
            + (
                f" -- skipped {len(stale_ops)} pruned op(s), backfilled from the "
                f"last real measurement: {', '.join(sorted(stale_ops))}"
                if stale_ops
                else ""
            ),
            logger="current",
        )

    def _probe_batch(self, model, data, snapshot, active_ops=None):
        """Returns ``(grads, magnitudes)`` -- ``grads`` is
        ``{op_name: (B,) array}`` for every op in ``active_ops`` only (the
        per-image ``d loss / d magnitude``, measured on a single shared clean
        batch so columns stay aligned across ops); ``magnitudes`` is the same
        shape but for ALL of `DIFF32_OPS`, always, regardless of `active_ops` --
        see the comment at its call site for why the magnitude draw must never
        be filtered.

        `snapshot` is the SA-published per-op magnitude distribution, or ``{}``
        for the fixed-magnitude fallback. `active_ops` is normally all of
        `DIFF32_OPS` -- it excludes whatever `_sweep` decided to skip this sweep
        under `skip_pruned`, which is where the actual forward+backward savings
        come from (skipped ops never reach `_grad_for_op`, backfilled by the
        caller from a cached prior measurement instead).

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

        batch_size = int(rgb01.shape[0])
        # Magnitudes for EVERY op, never filtered to active_ops -- the sampled
        # modes draw from self._magnitude_rng per op, in op_names iteration
        # order (see sample_magnitudes), so filtering this list would shift
        # which draw lands on which MEASURED op depending on what else happens
        # to be pruned that round. Computing them is cheap (no model pass); only
        # the forward+backward below is worth skipping.
        if self.per_image_delta:
            magnitudes = resolve_magnitudes(
                snapshot,
                list(DIFF32_OPS),
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
                for name in DIFF32_OPS
            }

        grads = {
            name: self._grad_for_op(
                name, model, op, rgb01, mean, std, data_samples, magnitudes[name]
            )
            for name, op in active_ops.items()
        }
        return grads, magnitudes

    def _grad_for_op(
        self, name, model, op, rgb01, mean, std, data_samples, magnitude
    ) -> np.ndarray:
        """d loss / d magnitude for one op, as a (B,) array -- one sensitivity
        per image. (In the legacy scalar mode, a (1,) array holding the old
        batch-averaged number.)

        `magnitude` is the (B,) array of probe magnitudes for this op, already
        resolved from the SA snapshot by the caller.

        Geometric ops warp the LABEL alongside the image. Without that, rotating
        an image while leaving its label map in place makes the loss report how far
        the warp moved pixels -- a fact about geometry, not about the model -- and
        every geometric column of R measures misalignment instead of sensitivity.
        Photometric ops move no pixels, so their labels are correct as they are.
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

        matrix = geometric_affine_matrix(name, rgb01, delta)
        if matrix is None:
            perturbed01 = op(rgb01, delta)
            warped_labels = None
        else:
            # warp_image_and_label reproduces the op's own image warp exactly (both
            # are `warp_affine(images, matrix)`), so calling it INSTEAD of `op` --
            # not in addition to it -- leaves the image path unchanged and the
            # gradient intact, while carrying the label along.
            perturbed01, warped_labels = warp_image_and_label(
                rgb01, _gather_gt(data_samples, name), matrix
            )

        # Re-normalize back into the model's expected (normalized) input space.
        perturbed = (perturbed01 * 255.0 - mean) / std

        with _swapped_gt(data_samples, warped_labels):
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
    """Collapse one op's per-image probe magnitudes over a sweep into a summary.

    Mean/min/max rather than the raw array: the per-image values are already
    written in full to aug_gradient_log.txt, and repeating 500 of them per op in
    the correlation record would bury R. For the constant modes (`mode`, `fixed`)
    min == mean == max, which is itself the check that the mode did what it says.
    """
    flat = np.concatenate(values)
    return {
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


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
