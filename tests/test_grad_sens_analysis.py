"""Tests for the augmentation gradient cross-correlation matrix R.

The maths here is pure numpy, but the module imports the mmseg/mmengine
registries, so run these in the `sensaug` conda env on a compute node.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS
from sensaug.hooks.grad_sens_analysis import (
    PerturbationSensitivityAnalysisHookWithGradients,
    _jsonable,
    closure_null,
    cluster_bootstrap_ci,
    correlate,
    fdr_correct,
    merge_rank_buffers,
    shared_factor_loadings,
    shared_image_factor,
    stack_probe_buffer,
)

NAMES = list(DIFFERENTIABLE_PERTURBATIONS)
N_OPS = len(NAMES)


def _buffer(probes_per_op):
    """probes_per_op: list of (B,) arrays, reused for every op."""
    return {name: [p.copy() for p in probes_per_op] for name in NAMES}


def _confounded_grid(n_ops=N_OPS, n_images=800, seed=0):
    """Ops that are genuinely INDEPENDENT of one another, observed through a
    shared per-image scale.

    Two facts drive this: augmentation sensitivities are predominantly positive
    (perturbing an image almost always increases its loss), and a hard image has a
    big gradient for EVERY op. So a shared per-image factor multiplies rows that
    have a positive mean -- which manufactures correlation out of nothing.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=1.0, scale=0.15, size=(n_ops, n_images))
    difficulty = rng.uniform(0.5, 5.0, size=n_images)
    return base * difficulty


# --- building D over images --------------------------------------------------


def test_stack_probe_buffer_concatenates_over_images_not_probes():
    """The axis IS the fix. 5 probes of 4 images must give 20 columns to correlate
    over, not 5."""
    probes = [np.arange(4, dtype=float) + i for i in range(5)]
    d_grad, probe_ids = stack_probe_buffer(_buffer(probes), NAMES)

    assert d_grad.shape == (N_OPS, 20)
    assert probe_ids.shape == (20,)
    assert np.unique(probe_ids).size == 5
    # column j carries the probe it came from -- the bootstrap's cluster id
    np.testing.assert_array_equal(probe_ids[:4], [0, 0, 0, 0])
    np.testing.assert_array_equal(probe_ids[4:8], [1, 1, 1, 1])


def test_stack_probe_buffer_allows_ragged_batches():
    """A short final batch (drop_last=False) shortens every op's row equally, so
    the columns stay aligned."""
    probes = [np.zeros(4), np.zeros(4), np.zeros(2)]
    d_grad, probe_ids = stack_probe_buffer(_buffer(probes), NAMES)

    assert d_grad.shape == (N_OPS, 10)
    assert np.unique(probe_ids).size == 3


def test_stack_probe_buffer_rejects_misaligned_ops():
    """Column j must mean the same image for every op. If one op is missing a
    probe, correlating the rows silently compares different images -- so this is
    an assert, not a truncation."""
    buffer = _buffer([np.zeros(4), np.zeros(4)])
    buffer[NAMES[3]] = [np.zeros(4)]  # one op fell behind

    with pytest.raises(AssertionError, match="misaligned"):
        stack_probe_buffer(buffer, NAMES)


# --- correlation + guards ----------------------------------------------------


def test_correlate_recovers_identical_and_independent_rows():
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 500))
    d_grad[1] = d_grad[0] + 1e-6 * rng.normal(size=500)  # a redundant pair

    r, dropped = correlate(d_grad)

    assert dropped.size == 0
    assert r[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert r[2, 3] == pytest.approx(0.0, abs=0.15)


def test_correlate_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    r, _ = correlate(rng.normal(size=(N_OPS, 300)))

    assert r.shape == (N_OPS, N_OPS)
    np.testing.assert_allclose(r, r.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(r), np.ones(N_OPS), atol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_correlate_drops_a_constant_row_instead_of_emitting_a_silent_nan():
    """An op with no variance over the window has no correlation signal at all.
    np.corrcoef would divide by its zero std, emit a RuntimeWarning, and quietly
    scatter NaN through the matrix. Drop it explicitly instead, and make sure the
    kept submatrix comes back clean (filterwarnings turns the warning into a
    failure)."""
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 200))
    d_grad[2] = 0.7  # this op never moves

    r, dropped = correlate(d_grad)

    np.testing.assert_array_equal(dropped, [2])
    assert np.isnan(r[2]).all()  # whole row, diagonal included
    assert np.isnan(r[:, 2]).all()
    keep = [0, 1, 3]
    assert np.isfinite(r[np.ix_(keep, keep)]).all()


# --- the image-difficulty confound -------------------------------------------


def test_shared_image_scale_inflates_raw_r_and_normalization_removes_it():
    """The failure mode this whole normalization exists for.

    Correlating raw per-image gradients does not measure redundancy -- it measures
    image difficulty. These 14 ops are independent by construction, yet raw R
    calls almost every pair redundant, for exactly the same reason the old
    correlate-over-training-time R did: a factor shared by every row.
    """
    d_grad = _confounded_grid()
    off_diagonal = np.triu_indices(N_OPS, k=1)

    r_raw, _ = correlate(d_grad)
    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    # Raw R: "everything is redundant." Nothing is.
    assert r_raw[off_diagonal].mean() > 0.5

    # Normalized R: no structure, correctly. It sits on the closure baseline
    # rather than exactly 0 -- see closure_null.
    assert r_norm[off_diagonal].mean() == pytest.approx(closure_null(N_OPS), abs=0.05)


def test_shared_factor_loadings_expose_the_confound():
    """The headline diagnostic: if every op loads ~1.0 on the shared per-image
    factor, R is reading which images are hard, and nothing built on it is
    trustworthy."""
    d_grad = _confounded_grid()

    loadings = shared_factor_loadings(d_grad, shared_image_factor(d_grad))

    assert loadings.shape == (N_OPS,)
    assert (loadings > 0.9).all()


def test_normalization_preserves_a_real_correlation():
    """The normalization must remove the shared factor WITHOUT flattening genuine
    redundancy -- otherwise it would just be destroying the signal."""
    rng = np.random.default_rng(1)
    n_images = 800
    base = rng.normal(loc=1.0, scale=0.15, size=(N_OPS, n_images))
    base[5] = base[2] + 0.05 * rng.normal(size=n_images)  # ops 2 and 5 really are alike
    d_grad = base * rng.uniform(0.5, 5.0, size=n_images)

    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    assert r_norm[2, 5] > 0.8
    others = [i for i in range(N_OPS) if i not in (2, 5)]
    assert abs(r_norm[np.ix_(others, others)][np.triu_indices(len(others), 1)].mean()) < 0.2


def test_closure_null_is_the_baseline_not_zero():
    assert closure_null(14) == pytest.approx(-1.0 / 13)
    assert closure_null(1) == 0.0


# --- bootstrap + FDR ---------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_a_naive_column_bootstrap():
    """Columns inside one probe share a batch and a model state, so they are not
    independent draws. Resampling COLUMNS pretends there are N_images independent
    observations and returns intervals that are too tight; resampling whole PROBES
    tells the truth. Same function, different cluster ids."""
    rng = np.random.default_rng(0)
    n_probes, batch = 20, 8
    probe_ids = np.repeat(np.arange(n_probes), batch)

    # A probe-level effect shared by every column of that probe -- precisely the
    # clustering a column bootstrap ignores.
    probe_effect = rng.normal(size=(2, n_probes)).repeat(batch, axis=1)
    d_grad = probe_effect + rng.normal(scale=0.3, size=(2, n_probes * batch))

    lo_probe, hi_probe, _ = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=400)
    lo_col, hi_col, _ = cluster_bootstrap_ci(
        d_grad, np.arange(probe_ids.size), n_reps=400
    )

    assert (hi_probe[0, 1] - lo_probe[0, 1]) > (hi_col[0, 1] - lo_col[0, 1])


def test_bootstrap_ci_narrows_as_more_probes_are_collected():
    def ci_width(n_probes):
        rng = np.random.default_rng(0)
        n_images = n_probes * 4
        d_grad = rng.normal(size=(2, n_images))
        d_grad[1] = 0.6 * d_grad[0] + 0.8 * rng.normal(size=n_images)
        lo, hi, _ = cluster_bootstrap_ci(
            d_grad, np.repeat(np.arange(n_probes), 4), n_reps=300
        )
        return hi[0, 1] - lo[0, 1]

    assert ci_width(120) < ci_width(15)


def test_only_a_genuinely_correlated_pair_survives_fdr():
    """91 pairwise cells means several will look significant by chance. BH is what
    stops a pruning decision resting on one of them."""
    rng = np.random.default_rng(0)
    n_probes, batch = 40, 8
    n_images = n_probes * batch
    probe_ids = np.repeat(np.arange(n_probes), batch)

    d_grad = rng.normal(size=(N_OPS, n_images))
    d_grad[9] = 0.9 * d_grad[3] + np.sqrt(1 - 0.81) * rng.normal(size=n_images)

    _, _, p = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=500)
    q, survives = fdr_correct(p, alpha=0.05)

    assert survives[3, 9], "the injected redundant pair should survive FDR"
    assert q[3, 9] < 0.05
    upper = np.triu_indices(N_OPS, k=1)
    assert survives[upper].sum() <= 3, "too many null pairs slipped through BH"


def test_fdr_is_symmetric_and_ignores_the_diagonal():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(6, 6))
    p = (p + p.T) / 2

    q, survives = fdr_correct(p)

    np.testing.assert_allclose(q, q.T, equal_nan=True)
    assert np.isnan(np.diag(q)).all()
    assert not np.diag(survives).any()


# --- DDP ---------------------------------------------------------------------


def test_merge_rank_buffers_preserves_probe_boundaries():
    """Each rank probes its own shard, so rank 0's R would otherwise see only
    1/world_size of the images. Merged probes stay SEPARATE entries -- they are
    different images, and the bootstrap resamples on those boundaries."""
    rank0 = {name: [np.zeros(4), np.zeros(4)] for name in NAMES}
    rank1 = {name: [np.ones(4)] for name in NAMES}

    merged = merge_rank_buffers([rank0, rank1], NAMES)
    d_grad, probe_ids = stack_probe_buffer(merged, NAMES)

    assert len(merged[NAMES[0]]) == 3
    assert d_grad.shape == (N_OPS, 12)
    assert np.unique(probe_ids).size == 3


# --- serialization -----------------------------------------------------------


def test_jsonable_writes_nan_as_null():
    """R deliberately contains NaN (dropped ops). json.dumps would emit a bare
    `NaN` token, which python reads back but jq and pandas will not."""
    payload = _jsonable({"R": np.array([[1.0, np.nan], [np.nan, 1.0]]), "n": np.int64(4)})

    text = json.dumps(payload)
    assert "NaN" not in text
    assert json.loads(text)["R"] == [[1.0, None], [None, 1.0]]
    assert json.loads(text)["n"] == 4


# --- checkpoint bucketing ----------------------------------------------------


class _FakeRunner:
    def __init__(self, work_dir, buffer, max_iters=1000):
        self.cfg = types.SimpleNamespace(work_dir=str(work_dir))
        self.aug_grad_buffer = buffer
        self.max_iters = max_iters
        self.iter = 0


def _hook(**kwargs):
    kwargs.setdefault("checkpoints", (0.5, 1.0))
    kwargs.setdefault("bootstrap", False)
    kwargs.setdefault("n_min", 8)
    return PerturbationSensitivityAnalysisHookWithGradients(**kwargs)


def _fill(buffer, n_probes, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_probes):
        for name in NAMES:
            buffer[name].append(rng.normal(size=batch))


def test_r_is_emitted_only_when_a_checkpoint_is_crossed(tmp_path):
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook()

    # 40% through: window still open, nothing emitted, nothing thrown away.
    _fill(buffer, n_probes=3)
    runner.iter = 400
    hook.after_val_epoch(runner)
    assert not (tmp_path / "corr_matrix_log.txt").exists()
    assert len(buffer[NAMES[0]]) == 3, "buffer must keep accumulating mid-window"

    # crossing 50%: emit, then clear so the next window is one model state
    _fill(buffer, n_probes=3, seed=1)
    runner.iter = 500
    hook.after_val_epoch(runner)

    records = (tmp_path / "corr_matrix_log.txt").read_text().strip().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["checkpoint"] == 0.5
    assert record["n_images"] == 24  # 6 probes x 4 images
    assert record["n_probes"] == 6
    assert len(buffer[NAMES[0]]) == 0, "window must be cleared at a checkpoint"


def test_window_below_n_min_is_skipped_and_discarded(tmp_path):
    """Two data points always lie on a line, so the old '>= 2 probes' gate made
    corrcoef return exactly +/-1 for every pair and logged it as real. A window
    that is too small is dropped, NOT rolled into the next checkpoint -- that
    would pool two different model states."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(n_min=100)

    _fill(buffer, n_probes=2)  # 8 images, well under n_min
    runner.iter = 500
    hook.after_val_epoch(runner)

    assert not (tmp_path / "corr_matrix_log.txt").exists()
    assert len(buffer[NAMES[0]]) == 0, "an undersized window is discarded, not carried"


def test_record_carries_both_matrices_and_the_confound_diagnostic(tmp_path):
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(normalize_per_image=True)

    _fill(buffer, n_probes=6)
    runner.iter = 500
    hook.after_val_epoch(runner)

    record = json.loads((tmp_path / "corr_matrix_log.txt").read_text().strip())
    assert record["names"] == NAMES
    assert np.array(record["R_raw"], dtype=float).shape == (N_OPS, N_OPS)
    assert np.array(record["R_scalenorm"], dtype=float).shape == (N_OPS, N_OPS)
    assert len(record["shared_factor_loadings"]) == N_OPS
    assert record["dropped"] == []


def test_bootstrap_detail_goes_to_its_own_file(tmp_path):
    """91 cells x several fields per checkpoint would bury R if inlined."""
    buffer = {name: [] for name in NAMES}
    runner = _FakeRunner(tmp_path, buffer, max_iters=1000)
    hook = _hook(bootstrap=True, bootstrap_reps=50)

    _fill(buffer, n_probes=8)
    runner.iter = 500
    hook.after_val_epoch(runner)

    record = json.loads((tmp_path / "corr_bootstrap_log.txt").read_text().strip())
    assert record["cluster_level"] == "probe"
    assert len(record["cells"]) == N_OPS * (N_OPS - 1) // 2  # 91
    cell = record["cells"][0]
    assert {"i", "j", "r", "ci_lo", "ci_hi", "ci_width", "p", "q", "survives_fdr"} <= set(cell)
    # R itself stays in the other file, ungarbled by all of this
    assert "R_raw" in json.loads((tmp_path / "corr_matrix_log.txt").read_text().strip())
