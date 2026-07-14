"""
Hook that computes the cross-correlation matrix R between the differentiable
augmentations, from the per-image gradient traces collected by
CollectGradientHook.

R is correlated across IMAGES, not across training time. CollectGradientHook
stores, per probe, one ``d loss / d magnitude`` value PER IMAGE per op, so
concatenating a window of probes gives a (A ops x N images) matrix whose rows can
be correlated to answer "do these two augmentations hurt the SAME images" -- that
is what redundancy means. Correlating across probes instead (one batch-averaged
number per probe) would only ask "do these two augmentations get less sensitive at
the same MOMENTS during training", and since nearly every op's sensitivity drifts
down together as the model converges, that reports almost everything as redundant.
It measures convergence drift, not redundancy.

The rows also carry a shared per-image scale factor -- hard images have a large
gradient for EVERY op, and mmseg's ignore_index weighting adds a per-image
valid-pixel factor on top. Pearson is invariant to a global scale but not to a
per-column one, so that factor would re-inflate R just as convergence drift did.
It is divided out (``normalize_per_image``), and its per-op loadings are logged at
every emission so you can see how much of R it was accounting for.

R is emitted every ``interval`` train iters from the observations collected since
the previous emission, never pooled across them: a warmup window and a converged
window describe different models, and pooling groups with different underlying
structure manufactures trends neither group has (Simpson's paradox).

That ``interval`` is the correlation pipeline's OWN clock, shared with
CollectGradientHook and with nothing else. Both hooks used to fire from
``after_val_epoch``, which under ``--aug-type=ours`` is called by RobustValLoop
itself, so R was computed inside the sensitivity-analysis pipeline's val epoch and
could only be emitted on an SA round. The two measure different things -- SA asks
which perturbations the model is worst at, R asks which perturbations are
redundant with each other -- so they now keep separate clocks and neither is
reachable from the other. See sensaug/loops.py for the SA pipeline, which this
module deliberately does not touch.

Scope: R covers the differentiable ops only. Pruning based on R is future work --
``prune_augmentations`` is a stub, pending sign-off on what "redundant" is allowed
to mean here.
"""

import os
import json
import math
import warnings

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process, get_world_size, all_gather_object
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS
from sensaug.hooks.grad_hook import fires_at, training_progress

# A row whose sensitivity never varies across the window carries no correlation
# signal; np.corrcoef would divide by its zero std and emit a silent NaN row.
_STD_EPS = 1e-12
# Guards the per-image scale division.
_DIV_EPS = 1e-12
# Above this, the shared image-difficulty factor is eating R.
_LOADING_ALARM = 0.9


def stack_probe_buffer(buffer: dict, names: list):
    """Flatten a window of probes into a (A, N_images) matrix.

    Returns (d_grad, probe_ids), where probe_ids[j] is the index of the probe
    column j came from -- the cluster id the bootstrap resamples on.

    Column alignment across ops holds because every op is probed on the same
    shared batch each probe, appended in the same op-loop order, and a failed
    probe is all-or-nothing. That is an invariant, so assert it rather than
    truncating to the shortest row (which would silently misalign the columns).
    """
    rows = [
        np.concatenate(buffer[name]) if buffer[name] else np.empty(0) for name in names
    ]
    lengths = {row.shape[0] for row in rows}
    assert len(lengths) == 1, (
        f"augmentation rows are misaligned ({sorted(lengths)}): every op must be "
        f"probed on the same batch, in the same order, on every probe"
    )

    reference = buffer[names[0]]
    probe_ids = (
        np.concatenate(
            [np.full(probe.shape[0], i, dtype=int) for i, probe in enumerate(reference)]
        )
        if reference
        else np.empty(0, dtype=int)
    )
    return np.stack(rows), probe_ids


def shared_image_factor(d_grad: np.ndarray) -> np.ndarray:
    """The per-image mean absolute sensitivity: the empirical "image difficulty"
    factor that every op loads on (hard image -> big gradient for everything)."""
    return np.abs(d_grad).mean(axis=0)


def shared_factor_loadings(d_grad: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """corr(each op's row, the shared factor).

    THE diagnostic for this matrix: if these are all ~0.9+, R is measuring which
    images are hard, not which augmentations are redundant, and nothing
    downstream of it is trustworthy.
    """
    loadings = np.full(d_grad.shape[0], np.nan)
    if shared.size < 2 or shared.std() < _STD_EPS:
        return loadings
    for i, row in enumerate(d_grad):
        if row.std() >= _STD_EPS:
            loadings[i] = np.corrcoef(row, shared)[0, 1]
    return loadings


def correlate(d_grad: np.ndarray):
    """Pearson correlation across the rows of a (A, N) matrix, with the
    zero-variance rows dropped rather than silently NaN'd.

    Returns (R, dropped_idx). R is always (A, A); a dropped op's row and column
    (diagonal included) are NaN, so a dropped op can never be mistaken for an
    uncorrelated one.
    """
    n_ops = d_grad.shape[0]
    r = np.full((n_ops, n_ops), np.nan)

    stds = d_grad.std(axis=1) if d_grad.shape[1] else np.zeros(n_ops)
    keep = np.flatnonzero(stds >= _STD_EPS)
    dropped = np.flatnonzero(stds < _STD_EPS)

    if keep.size >= 2:
        # Only the kept rows go in, so corrcoef never divides by zero and never
        # emits a RuntimeWarning.
        sub = np.corrcoef(d_grad[keep])
        assert np.isfinite(sub).all(), "corrcoef produced a non-finite kept cell"
        r[np.ix_(keep, keep)] = sub

    return r, dropped


def closure_null(n_ops: int) -> float:
    """The baseline correlation induced purely by the per-image normalization.

    Dividing every row by a normalizer ESTIMATED FROM THOSE SAME ROWS closes the
    composition: the A normalized values in a column are tied together, which
    pushes the pairwise correlations to about -1/(A-1) even when the underlying
    ops are perfectly independent. At A=14 that is -0.077.

    Small, but it is the null: cells must be judged against THIS, not against
    zero, or every mildly negative cell reads as significant. (An external
    normalizer -- e.g. the clean per-image loss, one extra forward per probe --
    would avoid the artifact entirely; that is the clean fix if -0.077 ever
    starts to matter.)
    """
    return -1.0 / (n_ops - 1) if n_ops > 1 else 0.0


def cluster_bootstrap_ci(
    d_grad: np.ndarray,
    probe_ids: np.ndarray,
    n_reps: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    null: float = 0.0,
):
    """Bootstrap CIs for every cell of R by resampling whole PROBES.

    Resampling probes, not individual image columns, is the point. Columns within
    a probe come from one batch scored by one model state, so they are not
    independent draws; a naive column bootstrap treats N_images as the sample
    size, understates the variance, and hands back intervals that are too tight.

    Returns (ci_lo, ci_hi, p). The interval is the percentile interval over the
    replicates. The p-value tests r == `null` (0 for a raw R, closure_null(A) for
    a per-image-normalized one) via a normal approximation on the bootstrap
    standard error.

    The p-value deliberately does NOT come from the replicate percentiles. A
    percentile p-value cannot go below 1/n_reps, and Benjamini-Hochberg over the
    91 pairwise cells needs p <= alpha/91 ~ 5.5e-4 for the very first rejection:
    at n_reps=1000 the floor is 1e-3, so nothing could EVER survive FDR and the
    correction would silently reject every cell. The bootstrap SE has no such
    floor and still carries the clustering.
    """
    rng = np.random.default_rng(seed)
    probes = np.unique(probe_ids)
    columns_by_probe = [np.flatnonzero(probe_ids == p) for p in probes]
    n_ops = d_grad.shape[0]

    observed, _ = correlate(d_grad)

    reps = np.full((n_reps, n_ops, n_ops), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_reps):
            picks = rng.integers(0, len(columns_by_probe), size=len(columns_by_probe))
            columns = np.concatenate([columns_by_probe[i] for i in picks])
            reps[k], _ = correlate(d_grad[:, columns])

        ci_lo = np.nanpercentile(reps, 100 * alpha / 2, axis=0)
        ci_hi = np.nanpercentile(reps, 100 * (1 - alpha / 2), axis=0)
        standard_error = np.nanstd(reps, axis=0)

    z = (observed - null) / np.maximum(standard_error, 1e-12)
    p = 2 * norm.sf(np.abs(z))
    return ci_lo, ci_hi, p


def fdr_correct(p: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg across the A*(A-1)/2 unique off-diagonal cells.

    With 14 ops that is 91 simultaneous tests, several of which will look
    significant by chance -- correcting for that is what stops a pruning decision
    resting on noise. Returns (q, survives), both (A, A) and symmetric.
    """
    n_ops = p.shape[0]
    upper = np.triu_indices(n_ops, k=1)
    flat = p[upper]

    q_flat = np.full(flat.shape, np.nan)
    survives_flat = np.zeros(flat.shape, dtype=bool)
    finite = np.isfinite(flat)
    if finite.any():
        rejected, q_values, _, _ = multipletests(
            flat[finite], alpha=alpha, method="fdr_bh"
        )
        q_flat[finite] = q_values
        survives_flat[finite] = rejected

    q = np.full((n_ops, n_ops), np.nan)
    survives = np.zeros((n_ops, n_ops), dtype=bool)
    q[upper] = q_flat
    q.T[upper] = q_flat
    survives[upper] = survives_flat
    survives.T[upper] = survives_flat
    return q, survives


def merge_rank_buffers(buffers: list, names: list) -> dict:
    """Concatenate the per-rank probe buffers into one window.

    Under DDP each rank probes its own data shard, so rank 0's R would otherwise
    see only 1/world_size of the images while every other rank's probe compute is
    thrown away. Rank r's probe p and rank r'-'s probe p are different images, so
    they stay separate entries -- probe boundaries (the bootstrap's clusters) are
    preserved.
    """
    merged = {name: [] for name in names}
    for buffer in buffers:
        for name in names:
            merged[name].extend(buffer.get(name, []))
    return merged


def _jsonable(value):
    """numpy -> python, and non-finite -> None.

    json.dumps writes a bare `NaN` token for float('nan'), which is not valid
    JSON: python reads it back, jq and pandas do not. R deliberately contains
    NaN (dropped ops), so every record goes through this on the way out.
    """
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_json_atomic(path, obj):
    """Rewrite `path` as one JSON document, via a temp file + os.replace.

    A whole-file rewrite can be interrupted halfway and leave a truncated file
    behind, where the append it replaced could not. os.replace is atomic, so a
    crash mid-write leaves the previous checkpoint's file fully intact rather than
    corrupting every checkpoint written so far.

    allow_nan=False so a value that skipped _jsonable raises here instead of
    writing a bare `NaN` token, which json.dump emits happily and which is not
    valid JSON (jq and pandas both reject it).
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


@HOOKS.register_module()
class PerturbationSensitivityAnalysisHookWithGradients(Hook):
    """Builds and logs the augmentation gradient cross-correlation matrix R.

    Consumes ``runner.aug_grad_buffer`` (populated by CollectGradientHook); no
    effect unless that hook is also registered.

    Args:
        interval (int): Train iters between emissions of R. Must match
            CollectGradientHook's ``interval`` -- the two are one pipeline on one
            clock. Each R uses only the probes collected since the previous
            emission; windows are never pooled, because a warmup model and a
            converged model are different measurement instruments.
        n_min (int): Minimum images in a window before R is computed at all. At
            n=100 the correlation standard error is ~1/sqrt(n-3) ~ 0.10. The old
            gate ("at least 2 probes") was meaningless: with 2 points corrcoef
            returns exactly +/-1 for every pair, since two points always lie on a
            line.
        normalize_per_image (bool): Divide out the shared per-image scale factor
            before correlating. On by default -- without it R re-confounds with
            image difficulty.
        bootstrap (bool): Compute cluster-bootstrap CIs and BH-FDR per cell.
        gather_ranks (bool): all_gather every rank's probes before building R.
    """

    def __init__(
        self,
        interval: int,
        n_min: int = 100,
        normalize_per_image: bool = True,
        bootstrap: bool = True,
        bootstrap_reps: int = 1000,
        bootstrap_seed: int = 0,
        fdr_alpha: float = 0.05,
        gather_ranks: bool = True,
    ) -> None:
        if interval < 1:
            raise ValueError(f"interval must be a positive iteration count, got {interval}")
        self.interval = interval
        self.n_min = n_min
        self.normalize_per_image = normalize_per_image
        self.bootstrap = bootstrap
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_seed = bootstrap_seed
        self.fdr_alpha = fdr_alpha
        self.gather_ranks = gather_ranks

        self.corr_log_path = None
        self.bootstrap_log_path = None
        self.names = list(DIFFERENTIABLE_PERTURBATIONS)

    def after_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        buffer = getattr(runner, "aug_grad_buffer", None)
        if buffer is None:
            return

        # The SAME gate as CollectGradientHook -- literally the same function -- at
        # the same hook point. The collector runs first (its NORMAL priority beats
        # this hook's LOW), so by the time we get here the buffer holds exactly the
        # sweep it just took: one frozen model state, which is the whole claim R
        # rests on.
        if not fires_at(runner, self.interval):
            return  # window still open -- keep accumulating

        checkpoint = training_progress(runner)

        # Gather before rank 0 builds R: each rank probed a different shard.
        window = buffer
        if self.gather_ranks and get_world_size() > 1:
            window = merge_rank_buffers(all_gather_object(buffer), self.names)

        if is_main_process():
            self._emit(runner, checkpoint, window)

        # Clear on EVERY rank (not just rank 0, or the other ranks' buffers grow
        # without bound), and only at an emission (so a window is one model state,
        # not a smear across several).
        for name in buffer:
            buffer[name].clear()

    def _emit(self, runner: Runner, checkpoint: float, buffer: dict) -> None:
        if self.corr_log_path is None:
            work_dir = runner.cfg.work_dir
            self.corr_log_path = os.path.join(work_dir, "corr_matrix_log.json")
            self.bootstrap_log_path = os.path.join(work_dir, "corr_bootstrap_log.txt")

        d_grad, probe_ids = stack_probe_buffer(buffer, self.names)
        n_images = int(d_grad.shape[1])
        n_probes = int(np.unique(probe_ids).size)

        if n_images < self.n_min:
            print_log(
                f"[grad-corr] checkpoint {checkpoint:.0%}: SKIPPED, window holds "
                f"only {n_images} images (n_min={self.n_min}). Discarding it "
                f"rather than rolling it into the next checkpoint, which would "
                f"pool two different model states.",
                logger="current",
            )
            return

        r_raw, dropped = self.calculate_cross_corelation(d_grad)
        shared = shared_image_factor(d_grad)
        loadings = shared_factor_loadings(d_grad, shared)

        # The matrix we would actually act on: de-confounded if enabled.
        d_primary = d_grad
        r_norm = None
        if self.normalize_per_image:
            d_primary = d_grad / (shared + _DIV_EPS)
            r_norm, dropped = self.calculate_cross_corelation(d_primary)
        r_primary = r_norm if r_norm is not None else r_raw

        dropped_names = [self.names[i] for i in dropped]
        record = {
            "checkpoint": checkpoint,
            "iter": int(runner.iter),
            "n_images": n_images,
            "n_probes": n_probes,
            "names": self.names,
            "R_raw": r_raw,
            "R_scalenorm": r_norm,
            "dropped": dropped_names,
            "shared_factor_loadings": loadings,
        }
        # Read-modify-write, so the file stays ONE json array across a run's
        # checkpoints (and across a resume, matching the append-only gradient log
        # -- see grad_hook.before_run). The array is at most one record per
        # checkpoint, so rewriting it whole costs nothing.
        records = []
        if os.path.exists(self.corr_log_path):
            with open(self.corr_log_path) as f:
                records = json.load(f)
        records.append(_jsonable(record))
        _write_json_atomic(self.corr_log_path, records)

        n_survivors = n_cells = None
        if self.bootstrap:
            n_survivors, n_cells = self._emit_bootstrap(
                checkpoint, d_primary, probe_ids, dropped
            )

        max_loading = np.nanmax(np.abs(loadings)) if np.isfinite(loadings).any() else np.nan
        summary = (
            f"[grad-corr] checkpoint {checkpoint:.0%} (iter {runner.iter}): "
            f"n_images={n_images} n_probes={n_probes} "
            f"dropped={dropped_names or 'none'} "
            f"max_shared_loading={max_loading:.2f}"
        )
        if n_cells:
            summary += f" fdr_survivors={n_survivors}/{n_cells}"
        print_log(summary, logger="current")

        self._log_to_tensorboard(
            runner,
            step=int(runner.iter),
            d_grad=d_grad,
            r_primary=r_primary,
            loadings=loadings,
            n_images=n_images,
            n_survivors=n_survivors,
            n_cells=n_cells,
        )

        if np.isfinite(max_loading) and max_loading > _LOADING_ALARM:
            print_log(
                f"[grad-corr] every op loads >{_LOADING_ALARM} on the shared "
                f"per-image factor: R is tracking which images are HARD, not "
                f"which augmentations are redundant. Do not act on it.",
                logger="current",
                level=30,  # WARNING
            )

        self.prune_augmentations(r_primary)

    def _emit_bootstrap(self, checkpoint, d_grad, probe_ids, dropped):
        """Per-cell CIs + BH-FDR, written to their own file -- 91 cells x several
        fields per checkpoint would bury R if inlined into corr_matrix_log.json.

        Stays append-only JSONL, unlike corr_matrix_log.json: it is the bulky one,
        and nothing needs to load it as a single document."""
        keep = np.setdiff1d(np.arange(len(self.names)), dropped)
        if keep.size < 2:
            return 0, 0

        d_keep = d_grad[keep]
        r, _ = correlate(d_keep)
        # The per-image normalization closes the composition, so an independent
        # pair sits at closure_null(A), not at 0. Testing against 0 would flag
        # every mildly negative cell as a discovery.
        null = closure_null(keep.size) if self.normalize_per_image else 0.0
        ci_lo, ci_hi, p = cluster_bootstrap_ci(
            d_keep,
            probe_ids,
            n_reps=self.bootstrap_reps,
            seed=self.bootstrap_seed,
            alpha=self.fdr_alpha,
            null=null,
        )
        q, survives = fdr_correct(p, alpha=self.fdr_alpha)

        cells = []
        for a in range(keep.size):
            for b in range(a + 1, keep.size):
                cells.append(
                    {
                        "i": self.names[keep[a]],
                        "j": self.names[keep[b]],
                        "r": r[a, b],
                        "ci_lo": ci_lo[a, b],
                        "ci_hi": ci_hi[a, b],
                        "ci_width": ci_hi[a, b] - ci_lo[a, b],
                        "p": p[a, b],
                        "q": q[a, b],
                        "survives_fdr": survives[a, b],
                    }
                )

        record = {
            "checkpoint": checkpoint,
            "n_reps": self.bootstrap_reps,
            "seed": self.bootstrap_seed,
            "cluster_level": "probe",
            "matrix": "R_scalenorm" if self.normalize_per_image else "R_raw",
            "null_r": null,
            "cells": cells,
        }
        with open(self.bootstrap_log_path, "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")

        upper = np.triu_indices(keep.size, k=1)
        return int(survives[upper].sum()), int(upper[0].size)

    def _log_to_tensorboard(
        self,
        runner,
        step: int,
        d_grad: np.ndarray,
        r_primary: np.ndarray,
        loadings: np.ndarray,
        n_images: int,
        n_survivors,
        n_cells,
    ) -> None:
        """Write cross-correlation summary data to TensorBoard.

        Uses mmengine's runner.visualizer, which routes to the
        TensorboardVisBackend configured in train.py.  All tags live under
        the ``grad_corr/`` prefix so they form their own section in the
        TensorBoard UI and do not collide with training or SA metrics.
        """
        vis = runner.visualizer

        # --- per-op scalar summaries ---
        mean_sens = np.abs(d_grad).mean(axis=1)  # (A,)
        for i, name in enumerate(self.names):
            tag = name.replace("_", "")
            vis.add_scalar(
                f"grad_corr/mean_sensitivity/{tag}", float(mean_sens[i]), step
            )
            if np.isfinite(loadings[i]):
                vis.add_scalar(
                    f"grad_corr/shared_loading/{tag}", float(loadings[i]), step
                )

        # --- aggregate scalars ---
        max_loading = (
            float(np.nanmax(np.abs(loadings)))
            if np.isfinite(loadings).any()
            else 0.0
        )
        vis.add_scalar("grad_corr/max_shared_loading", max_loading, step)
        vis.add_scalar("grad_corr/n_images", n_images, step)
        if n_survivors is not None and n_cells is not None:
            vis.add_scalar("grad_corr/fdr_survivors", n_survivors, step)
            vis.add_scalar("grad_corr/fdr_total_cells", n_cells, step)

        # --- top correlated pairs (off-diagonal of primary R) ---
        n_ops = r_primary.shape[0]
        pairs = []
        for a in range(n_ops):
            for b in range(a + 1, n_ops):
                val = r_primary[a, b]
                if np.isfinite(val):
                    pairs.append((abs(val), val, a, b))
        pairs.sort(reverse=True)
        for _rank, (_, val, a, b) in enumerate(pairs[:5]):
            tag_a = self.names[a].replace("_", "")
            tag_b = self.names[b].replace("_", "")
            vis.add_scalar(
                f"grad_corr/top_pair/{tag_a}_vs_{tag_b}", float(val), step
            )

        # --- R heatmap as an image ---
        self._log_r_heatmap(vis, step, r_primary)

    def _log_r_heatmap(self, vis, step: int, r: np.ndarray) -> None:
        """Render R as a matplotlib heatmap and log it as a TensorBoard image."""
        short_names = [
            n.replace("Transform", "").replace("_", "") for n in self.names
        ]
        n = len(short_names)

        fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
        # Mask NaN for display
        r_display = np.where(np.isfinite(r), r, 0.0)
        im = ax.imshow(r_display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = r[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.5 else "black",
                    )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Augmentation Gradient Cross-Correlation (R)", fontsize=9)
        fig.tight_layout()

        # Render to numpy array (H, W, 3)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        vis.add_image("grad_corr/R_matrix", img, step, dataformats="HWC")

    def calculate_cross_corelation(self, d_grad: np.ndarray):
        """Pearson cross-correlation across the augmentation rows of D_grad.

        Args:
            d_grad (np.ndarray): shape (A, N_images) -- A differentiable ops x
                N images in the window. The images axis is the point: correlating
                over probes instead would measure convergence drift.

        Returns:
            (np.ndarray, np.ndarray): the symmetric (A, A) correlation matrix,
            and the indices of ops dropped for having no variance over the window
            (their rows/cols are NaN).
        """
        return correlate(d_grad)

    def prune_augmentations(self, r: np.ndarray) -> None:
        """Prune redundant augmentations given the correlation matrix R.

        Deliberately still a stub. R defines "redundant" as loss-gradient
        alignment at a fixed reference magnitude -- not as overlap in held-out
        per-image performance drop, which is what the pruning claim actually
        wants to rest on. That definition needs sign-off before anything acts on
        it. When it does: soft down-weighting of the sampling probability only,
        never hard deletion -- at the correlation sizes expected here (~0.3-0.5)
        deletion is unjustified.
        """
        pass
