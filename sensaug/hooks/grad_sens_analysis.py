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
reachable from the other. See sensaug/loops/sensaug_loop.py for the SA pipeline, which this
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

from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import is_main_process, get_world_size, all_gather_object
from mmseg.registry import HOOKS
from mmengine.hooks import Hook

# Local imports
# The merged 32-op vocabulary -- these are the axes of R. Aliased to the old
# name so nothing else in this module changes.
from sensaug.dataset.differentiable_augmentations_aa import (
    DIFF32_OPS,
)
from sensaug.hooks.grad_hook import fires_at, training_progress
from sensaug.redundancy import MODES, compute_red, summarise
# Imported straight off the submodule, NOT as `from sensaug.hooks import corr_viz`.
# hooks/__init__.py imports this module, so the package form is a cycle through a
# half-initialized package -- and the `except ImportError: pass` wrapping that
# __init__ would swallow the failure silently, leaving both hooks unregistered and
# the run dying later on an opaque registry miss.
from sensaug.hooks.corr_viz import markdown_report, render_matrix

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


def _first_primary(records: list):
    """The primary R from the earliest record in corr_matrix_log.json, as an
    (A, A) float array -- the baseline for the drift panel.

    Reads the file rather than caching the first matrix on the hook so that a
    --resume keeps diffing against the run's true first emission instead of
    restarting the baseline at whatever the restart happened to measure.

    _jsonable wrote every NaN out as null, so None comes back in and has to be
    turned into NaN again; np.array(..., dtype=float) does that for None but not
    for a nested list containing it, hence the explicit conversion.
    """
    if not records:
        return None
    raw = records[0].get("R_scalenorm") or records[0].get("R_raw")
    if not raw:
        return None
    return np.array(
        [[np.nan if v is None else float(v) for v in row] for row in raw], dtype=float
    )


def _tb_writer(vis):
    """The raw torch SummaryWriter under mmengine's Visualizer, or None.

    mmengine's Visualizer only forwards add_scalar and add_image to its backends,
    so the HISTOGRAMS and TEXT tabs are unreachable through the documented API.
    The writer itself is one attribute deeper, on the backend.

    Everything here is best-effort and swallows failures: this runs inside
    after_train_iter, and a TensorBoard nicety must never be able to take a
    training run down. A None return degrades the pipeline to scalars + images.
    """
    try:
        backends = getattr(vis, "_vis_backends", None) or {}
        backend = None
        if hasattr(vis, "get_backend"):
            backend = vis.get_backend("TensorboardVisBackend")
        if backend is None:
            # Don't insist on the class name: the backend is registered under
            # whatever key train.py's visualizer config gave it.
            backend = next(
                (b for b in backends.values() if type(b).__name__.startswith("Tensorboard")),
                None,
            )
        if backend is None:
            return None
        writer = backend.experiment
        # Duck-type rather than isinstance: this only needs to be something that
        # can take a histogram and a string.
        if hasattr(writer, "add_histogram") and hasattr(writer, "add_text"):
            return writer
    except Exception:  # noqa: BLE001
        return None
    return None


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
        red_mode: str = "squared",
        mask_within_op: bool = True,
    ) -> None:
        if interval < 1:
            raise ValueError(f"interval must be a positive iteration count, got {interval}")
        if red_mode not in MODES:
            raise ValueError(f"unknown red_mode {red_mode!r}, expected one of {MODES}")
        self.interval = interval
        self.n_min = n_min
        self.normalize_per_image = normalize_per_image
        self.bootstrap = bootstrap
        self.bootstrap_reps = bootstrap_reps
        self.bootstrap_seed = bootstrap_seed
        self.fdr_alpha = fdr_alpha
        self.gather_ranks = gather_ranks
        # How the per-op redundancy score reduces a row of R. "squared" is
        # closure-proof and sign-convention independent; "abs" and "signed" are
        # ablation arms. See sensaug/redundancy.py.
        self.red_mode = red_mode
        self.mask_within_op = mask_within_op

        self.corr_log_path = None
        self.bootstrap_log_path = None
        self.redundancy_log_path = None
        self.names = list(DIFF32_OPS)

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

        # Gather before R is built: each rank probed a different shard.
        window = buffer
        if self.gather_ranks and get_world_size() > 1:
            window = merge_rank_buffers(all_gather_object(buffer), self.names)

        # EVERY rank builds R, not just rank 0. The gather above leaves all ranks
        # holding identical input and _emit is deterministic (fixed bootstrap_seed,
        # pure numpy), so they all arrive at the same redundancy score.
        #
        # That is load-bearing rather than tidy: RobustValLoop rebuilds each rank's
        # OWN train dataloader from that rank's OWN pdf, so a score published on
        # rank 0 alone left ranks 1..N-1 sampling an un-reweighted pdf. DDP averages
        # all of their gradients into one update, so the reweighting reached only
        # 1/world_size of it -- silently, since rank 0 is also the rank that logs.
        #
        # The rank-0 guards now sit on the side effects inside _emit (log files,
        # TensorBoard) instead of on the computation that feeds training.
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
        # Which magnitudes the sweep was taken at, published by CollectGradientHook.
        # Recorded with R because two matrices measured at different magnitudes are
        # NOT comparable -- an R built at the fixed 0.5 fallback and one built from
        # the SA distribution answer different questions, and without this field
        # they are indistinguishable in the log.
        magnitude_info = getattr(runner, "aug_grad_magnitude_info", None) or {}
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
            "magnitude_source": magnitude_info.get("source"),
            "magnitude_mode": magnitude_info.get("mode"),
            "magnitudes": magnitude_info.get("per_op"),
        }
        # Rank 0 owns the file. Every rank reaches here now, and a read-modify-write
        # from world_size processes against one path interleaves and drops records.
        # first_r stays None off rank 0: its only consumer is the TensorBoard drift
        # panel, which is rank-0 only for the same reason.
        first_r = None
        if is_main_process():
            # Read-modify-write, so the file stays ONE json array across a run's
            # checkpoints (and across a resume, matching the append-only gradient
            # log -- see grad_hook.before_run). The array is at most one record per
            # checkpoint, so rewriting it whole costs nothing.
            records = []
            if os.path.exists(self.corr_log_path):
                with open(self.corr_log_path) as f:
                    records = json.load(f)
            # The run's FIRST R, for the drift panel -- taken from the file rather
            # than cached on self, so a --resume still diffs against the true first
            # emission of the run instead of the first one after the restart.
            first_r = _first_primary(records)
            records.append(_jsonable(record))
            _write_json_atomic(self.corr_log_path, records)

        n_survivors = n_cells = None
        cell_stats = None
        if self.bootstrap:
            n_survivors, n_cells, cell_stats = self._emit_bootstrap(
                checkpoint, d_primary, probe_ids, dropped
            )

        max_loading = np.nanmax(np.abs(loadings)) if np.isfinite(loadings).any() else np.nan
        summary = (
            f"[grad-corr] checkpoint {checkpoint:.0%} (iter {runner.iter}): "
            f"n_images={n_images} n_probes={n_probes} "
            f"dropped={dropped_names or 'none'} "
            f"max_shared_loading={max_loading:.2f} "
            f"magnitudes={magnitude_info.get('source', 'unknown')}/"
            f"{magnitude_info.get('mode', 'unknown')}"
        )
        if n_cells:
            summary += f" fdr_survivors={n_survivors}/{n_cells}"

        # Reporting only. Every rank computed the same numbers, so letting all of
        # them log would print world_size identical lines and write world_size sets
        # of TensorBoard scalars under the same tags into the same event directory.
        if is_main_process():
            print_log(summary, logger="current")

            self._log_to_tensorboard(
                runner,
                step=int(runner.iter),
                checkpoint=checkpoint,
                d_grad=d_grad,
                r_primary=r_primary,
                r_raw=r_raw,
                first_r=first_r,
                loadings=loadings,
                dropped_names=dropped_names,
                n_images=n_images,
                n_probes=n_probes,
                n_survivors=n_survivors,
                n_cells=n_cells,
                cell_stats=cell_stats,
                magnitude_info=magnitude_info,
            )

        if np.isfinite(max_loading) and max_loading > _LOADING_ALARM:
            print_log(
                f"[grad-corr] every op loads >{_LOADING_ALARM} on the shared "
                f"per-image factor: R is tracking which images are HARD, not "
                f"which augmentations are redundant. Do not act on it.",
                logger="current",
                level=30,  # WARNING
            )

        self.prune_augmentations(
            runner,
            r_primary,
            checkpoint=checkpoint,
            survives=(cell_stats or {}).get("survives"),
            max_loading=max_loading,
        )

    def _emit_bootstrap(self, checkpoint, d_grad, probe_ids, dropped):
        """Per-cell CIs + BH-FDR, written to their own file -- 91 cells x several
        fields per checkpoint would bury R if inlined into corr_matrix_log.json.

        Stays append-only JSONL, unlike corr_matrix_log.json: it is the bulky one,
        and nothing needs to load it as a single document.

        Returns ``(n_survivors, n_cells, stats)``, where ``stats`` is a dict of
        (A, A) arrays -- ci_lo, ci_hi, q, survives -- in the CANONICAL op index
        space, or None when there was nothing to test. The re-expansion matters:
        everything below is computed on ``d_keep``, so its indices count kept ops,
        not ops. Handing those arrays to the figure unexpanded would label every
        cell correctly right up until the first op is dropped, and then silently
        mislabel all of them.
        """
        keep = np.setdiff1d(np.arange(len(self.names)), dropped)
        if keep.size < 2:
            return 0, 0, None

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
        # The CIs above are computed on every rank -- `survives` gates the
        # redundancy score, so every rank needs it -- but only rank 0 appends, or
        # the file collects world_size copies of every record.
        if is_main_process():
            with open(self.bootstrap_log_path, "a") as f:
                f.write(json.dumps(_jsonable(record)) + "\n")

        upper = np.triu_indices(keep.size, k=1)
        return (
            int(survives[upper].sum()),
            int(upper[0].size),
            self._expand_to_ops(keep, ci_lo=ci_lo, ci_hi=ci_hi, q=q, survives=survives),
        )

    def _expand_to_ops(self, keep: np.ndarray, **arrays) -> dict:
        """Scatter kept-op-indexed (K, K) arrays back into (A, A) op space.

        Boolean arrays fill with False (a cell that was never tested did not
        survive), everything else with NaN (it has no value, and NaN is what the
        rest of this module already uses for "not measured").
        """
        n_ops = len(self.names)
        index = np.ix_(keep, keep)
        expanded = {}
        for name, array in arrays.items():
            if array.dtype == bool:
                full = np.zeros((n_ops, n_ops), dtype=bool)
            else:
                full = np.full((n_ops, n_ops), np.nan)
            full[index] = array
            expanded[name] = full
        return expanded

    def _log_to_tensorboard(
        self,
        runner,
        step: int,
        checkpoint: float,
        d_grad: np.ndarray,
        r_primary: np.ndarray,
        r_raw: np.ndarray,
        first_r,
        loadings: np.ndarray,
        dropped_names,
        n_images: int,
        n_probes: int,
        n_survivors,
        n_cells,
        cell_stats=None,
        magnitude_info=None,
    ) -> None:
        """Write R and its diagnostics to TensorBoard.

        Scalars and images go through mmengine's ``runner.visualizer``, which
        forwards to the TensorboardVisBackend configured in train.py. The
        histogram and the text report go through the raw SummaryWriter underneath
        it, which mmengine's Visualizer does not expose -- see ``_tb_writer``.

        Every tag lives under ``grad_corr/`` so the pipeline owns its own section
        of the UI and cannot collide with training or SA metrics.

        TAG STABILITY IS THE POINT. Every tag emitted here is a pure function of
        the op names, never of this emission's values, so each one is a continuous
        curve across the run's checkpoints. The previous ``top_pair/<a>_vs_<b>``
        tags were built from whichever 5 pairs happened to rank highest at that
        emission, so the tag SET moved between emissions and a four-checkpoint run
        produced up to twenty one-point series -- nothing that could be compared.
        """
        vis = getattr(runner, "visualizer", None)
        if vis is None:
            return

        magnitude_info = magnitude_info or {}
        max_loading = (
            float(np.nanmax(np.abs(loadings))) if np.isfinite(loadings).any() else 0.0
        )

        # --- per-op scalars ---
        # Op names go into tags verbatim. The old code stripped underscores, which
        # bought nothing (TensorBoard tags take them) and would silently merge any
        # future `color_jitter` into a `colorjitter`.
        mean_sens = np.abs(d_grad).mean(axis=1)  # (A,)
        for i, name in enumerate(self.names):
            vis.add_scalar(f"grad_corr/mean_sensitivity/{name}", float(mean_sens[i]), step)
            if np.isfinite(loadings[i]):
                vis.add_scalar(f"grad_corr/shared_loading/{name}", float(loadings[i]), step)

        # The magnitude each op was probed at, as its own series. This is what makes
        # a jump in R readable: a cell moving because the model changed looks the
        # same as one moving because the SA distribution shifted the magnitude under
        # it, and only these curves separate the two.
        for name, summary in magnitude_info.get("per_op", {}).items():
            vis.add_scalar(f"grad_corr/magnitude/{name}", float(summary["mean"]), step)

        # --- every pair, on a fixed tag ---
        # All A*(A-1)/2 of them, not a top-N: 91 scalar series is nothing for
        # TensorBoard, and it means `grad_corr/pair/blur` in the tag filter pulls
        # every pair involving blur across the whole run.
        upper = np.triu_indices(len(self.names), k=1)
        for a, b in zip(*upper):
            value = r_primary[a, b]
            if np.isfinite(value):
                vis.add_scalar(
                    f"grad_corr/pair/{self.names[a]}__{self.names[b]}",
                    float(value),
                    step,
                )

        # --- aggregates ---
        offdiag = r_primary[upper]
        finite = offdiag[np.isfinite(offdiag)]
        vis.add_scalar("grad_corr/max_shared_loading", max_loading, step)
        vis.add_scalar("grad_corr/n_images", n_images, step)
        vis.add_scalar("grad_corr/summary/n_probes", n_probes, step)
        vis.add_scalar("grad_corr/summary/n_dropped", len(dropped_names), step)
        if finite.size:
            vis.add_scalar(
                "grad_corr/summary/mean_abs_offdiag", float(np.abs(finite).mean()), step
            )
            vis.add_scalar(
                "grad_corr/summary/max_abs_offdiag", float(np.abs(finite).max()), step
            )
            vis.add_scalar(
                "grad_corr/summary/frac_abs_gt_0p5",
                float((np.abs(finite) > 0.5).mean()),
                step,
            )
        if n_survivors is not None and n_cells is not None:
            vis.add_scalar("grad_corr/fdr_survivors", n_survivors, step)
            vis.add_scalar("grad_corr/fdr_total_cells", n_cells, step)

        # --- heatmaps ---
        subtitle = (
            f"checkpoint {checkpoint:.0%} · iter {step} · {n_images} images / "
            f"{n_probes} probes · "
            f"{magnitude_info.get('source', 'unknown')}/"
            f"{magnitude_info.get('mode', 'unknown')} · "
            f"max shared loading {max_loading:.2f}"
        )
        # A matrix the shared-factor alarm fires on gets the warning stamped ON the
        # figure. The print_log alarm scrolls away in a SLURM log; a screenshot of
        # the heatmap does not, and the heatmap is what gets pasted into a slide.
        warn = (
            f"max shared-factor loading {max_loading:.2f} > {_LOADING_ALARM} — R is "
            f"tracking which images are HARD, not which augs are redundant. "
            f"Do not act on it."
            if max_loading > _LOADING_ALARM
            else None
        )
        # Only the FDR survivors get a printed number, so the cells you may act on
        # stop competing with ~180 you may not. Without a bootstrap there is no
        # survivorship to mark, and nothing is annotated.
        mark = cell_stats["survives"] if cell_stats else None

        vis.add_image(
            "grad_corr/R/scalenorm" if self.normalize_per_image else "grad_corr/R/raw",
            render_matrix(
                r_primary, self.names,
                "Augmentation gradient cross-correlation"
                + (" (scale-normalized)" if self.normalize_per_image else " (raw)"),
                subtitle=subtitle, mark=mark, warn=warn,
                cbar_label="Pearson r",
            ),
            step,
        )
        if self.normalize_per_image:
            # The un-normalized matrix beside the normalized one is how you see how
            # much of R was the shared per-image difficulty factor rather than
            # augmentation redundancy. Skipped when normalization is off, because
            # then it IS the primary matrix and was just logged above.
            vis.add_image(
                "grad_corr/R/raw",
                render_matrix(
                    r_raw, self.names,
                    "Augmentation gradient cross-correlation (raw, confounded)",
                    subtitle=subtitle + " · NOT de-confounded",
                    cbar_label="Pearson r",
                ),
                step,
            )

        if first_r is not None and first_r.shape == r_primary.shape:
            # Drift since the run's first emission. Same fixed +/-1 scale as the
            # absolute panels, so a faint delta panel means "R barely moved" rather
            # than "the colours got rescaled".
            vis.add_image(
                "grad_corr/R/delta_vs_first",
                render_matrix(
                    r_primary - first_r, self.names,
                    "Drift in R since this run's first emission",
                    subtitle=subtitle, cbar_label="Δ Pearson r",
                ),
                step,
            )

        self._log_tb_extras(
            vis, step, checkpoint, finite, r_primary, dropped_names,
            n_images, n_probes, max_loading, cell_stats, magnitude_info,
        )

    def _log_tb_extras(
        self, vis, step, checkpoint, offdiag_finite, r_primary, dropped_names,
        n_images, n_probes, max_loading, cell_stats, magnitude_info,
    ) -> None:
        """The histogram and text tabs, which need the raw SummaryWriter.

        Entirely optional: if the writer is unreachable the scalars and images
        above have already landed, and this is the part that is allowed to be
        missing.
        """
        writer = _tb_writer(vis)
        if writer is None:
            return

        stats = cell_stats or {}
        try:
            if offdiag_finite.size:
                # The whole redundancy DISTRIBUTION per checkpoint. TensorBoard
                # stacks these into a ridge, so "is the vocabulary getting more
                # redundant as the model converges" is one view rather than four
                # heatmaps read side by side.
                writer.add_histogram("grad_corr/offdiag_r", offdiag_finite, step)
            writer.add_text(
                "grad_corr/report",
                markdown_report(
                    self.names, r_primary,
                    checkpoint=checkpoint, iteration=step,
                    n_images=n_images, n_probes=n_probes,
                    dropped_names=dropped_names,
                    magnitude_source=magnitude_info.get("source"),
                    magnitude_mode=magnitude_info.get("mode"),
                    max_shared_loading=max_loading,
                    ci_lo=stats.get("ci_lo"), ci_hi=stats.get("ci_hi"),
                    q=stats.get("q"), survives=stats.get("survives"),
                ),
                step,
            )
        except Exception as e:  # noqa: BLE001
            print_log(
                f"[grad-corr] histogram/text logging skipped: {e}",
                logger="current",
                level=30,  # WARNING
            )

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

    def prune_augmentations(
        self,
        runner: Runner,
        r: np.ndarray,
        checkpoint: float = None,
        survives: np.ndarray = None,
        max_loading: float = np.nan,
    ) -> None:
        """Publish a per-op redundancy score for the SA loop to reweight the pdf by.

        Soft down-weighting of the sampling probability only, never hard deletion:
        at the correlation sizes actually observed here (mean |r| of 0.11-0.22)
        deletion would not be justified, and exp() is strictly positive so it is
        structurally impossible rather than merely avoided. The scoring itself lives
        in sensaug/redundancy.py, deliberately free of mmseg so it is testable
        without a GPU.

        Named for what it used to be a stub for. It does not prune.

        Published onto the runner rather than pushed, the same handoff channel
        `runner.corr_magnitudes` uses in the other direction: this hook fires on
        `corr_interval` (4 emissions by default) and the SA loop regenerates the pdf
        on `round_interval` (20 rounds), so the loop reads whatever is current and
        the two clocks never have to agree. Before the first emission there is no
        score and the pdf is untouched.

        The score is a per-op summary of R, and R defines "redundant" as loss-
        gradient alignment at the probe magnitude -- not as overlap in held-out
        per-image performance drop, which is what a pruning claim would need to rest
        on. Down-weighting is defensible on the weaker definition; deletion is not.
        """
        if not np.isfinite(r).any():
            return

        # The shared-factor alarm already prints "Do not act on it" a few lines
        # above; until now nothing enforced it. When every op loads on one
        # per-image factor, R is ranking which IMAGES are hard, and a score built
        # from it would reweight augmentations on the strength of image difficulty.
        if np.isfinite(max_loading) and max_loading > _LOADING_ALARM:
            runner.corr_redundancy = None
            print_log(
                f"[grad-corr] redundancy score withheld: max shared-factor loading "
                f"{max_loading:.2f} > {_LOADING_ALARM}. The pdf is left unweighted.",
                logger="current",
                level=30,  # WARNING
            )
            return

        # No bootstrap means no survivor mask; score every cell rather than fail.
        # Losing the multiplicity gate makes the score noisier, not wrong.
        score = compute_red(
            r,
            self.names,
            mode=self.red_mode,
            mask_within_op=self.mask_within_op,
            survivor_mask=survives,
        )

        if not score.usable:
            runner.corr_redundancy = None
            print_log(
                f"[grad-corr] redundancy score unusable, pdf left unweighted: "
                f"{score.reason}",
                logger="current",
                level=30,  # WARNING
            )
            return

        runner.corr_redundancy = {
            "iter": int(getattr(runner, "iter", 0)),
            "checkpoint": checkpoint,
            "mode": score.mode,
            "fdr_gated": survives is not None,
            "red": score.as_dict(),
            "raw": {n: float(v) for n, v in zip(score.names, score.raw)},
            "dropped": score.dropped,
            # R itself, plus the two masks that were applied to it. A down-weighting
            # method built on a per-op SUMMARY of R needs only "red" above, but a
            # PAIRWISE one -- mRMR, clustering, an eigenvector projection -- needs
            # the matrix, and cannot reconstruct it from the row sums. The record is
            # handed to those methods whole (see DOWNWEIGHT_METHODS in
            # sensaug/loops/grad_corr_loop.py), so carrying the structure here is
            # what lets a new arm be a function rather than a signature change.
            #
            # `names` is the axis order of BOTH `r` and `survives`; `mode` and
            # `mask_within_op` say how compute_red read them, so a pairwise method
            # can honour --corr-red-mode and --corr-keep-within-op identically
            # instead of quietly disagreeing with red(a) about what R says.
            "names": list(self.names),
            "mask_within_op": bool(self.mask_within_op),
            "r": np.asarray(r, dtype=np.float64),
            "survives": survives,
        }

        print_log(f"[grad-corr] {summarise(score)}", logger="current")

        if not is_main_process():
            return

        if self.redundancy_log_path is None:
            self.redundancy_log_path = os.path.join(
                runner.cfg.work_dir, "corr_redundancy_log.txt"
            )
        # JSONL append, matching corr_bootstrap_log.txt: one record per emission,
        # so the score is recomputable offline and a resume does not rewrite
        # history.
        #
        # `r` and `survives` are dropped on the way out. They exist on the runner
        # record for the pairwise down-weighting methods, but both are already
        # written in full to corr_matrix_log.json / corr_bootstrap_log.txt under the
        # same `iter`, so writing them a third time would turn this file's one line
        # per emission into an AxA block and buy nothing: join on `iter` instead.
        with open(self.redundancy_log_path, "a") as f:
            logged = {
                k: v
                for k, v in runner.corr_redundancy.items()
                if k not in ("r", "survives")
            }
            f.write(json.dumps(_jsonable(logged)) + "\n")
