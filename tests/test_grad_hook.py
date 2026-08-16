"""Tests for CollectGradientHook's frozen-frame per-image gradient sweep.

Requires the full mmseg/mmengine stack (grad_hook imports the registries), so run
these in the `sensaug` conda env on a compute node, not on a laptop.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mmengine")
pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The merged 32-op vocabulary -- what CollectGradientHook actually sweeps.
from sensaug.dataset.differentiable_augmentations_aa import (
    DIFF32_OPS as DIFFERENTIABLE_PERTURBATIONS,
)
from sensaug.hooks import grad_hook
from sensaug.hooks.grad_hook import CollectGradientHook, ProbeError



class _FakePreprocessor:
    """mean=0, std=255 makes the hook's de-normalize/re-normalize round trip the
    identity, so the model sees exactly the [0, 1] images we hand it."""

    mean = torch.zeros(3)
    std = torch.full((3,), 255.0)

    def __call__(self, data, training=False):
        return data


class _TinySegModel(torch.nn.Module):
    """Minimal stand-in for an mmseg segmentor: the probe only needs
    ``.loss(inputs, data_samples)`` returning a dict of loss tensors, plus a
    ``.data_preprocessor``.

    BatchNorm is the entire point of this fixture. It is the module whose
    train/eval mode decides whether the images in a batch are coupled, and the
    per-image gradient measurement lives or dies on that.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.data_preprocessor = _FakePreprocessor()

    def forward(self, x):
        return self.bn(self.conv(x))

    def loss(self, inputs, data_samples=None):
        # A batch-mean loss, mirroring mmseg's CE with reduction='mean'.
        return {"loss_ce": self.forward(inputs).pow(2).mean()}


class _FakeRunner:
    def __init__(self, model, iteration=0, max_iters=100):
        self.model = model
        self.iter = iteration
        self.max_iters = max_iters


@pytest.fixture
def images():
    torch.manual_seed(1)
    return torch.rand(4, 3, 16, 16)


def _batch(images):
    return {"inputs": images, "data_samples": None}


def _sweep_hook(tmp_path, batches, names=None, **kwargs):
    """A hook wired to an in-memory probe loader -- `batches` is the list of
    batches the sweep will iterate, standing in for the val dataloader."""
    kwargs.setdefault("interval", 1)  # irrelevant to tests that call _sweep directly
    hook = CollectGradientHook(**kwargs)
    hook.grad_buffer = {name: [] for name in (names or DIFFERENTIABLE_PERTURBATIONS)}
    hook._probe_loader = batches
    hook.grad_log_path = str(tmp_path / "aug_gradient_log.txt")
    return hook


def _grad_wrt_delta(model, images, deltas, op):
    delta = deltas.clone().requires_grad_(True)
    loss = model.loss(op(images, delta))["loss_ce"]
    return torch.autograd.grad(loss, delta)[0]


# --- the premise: per-image independence -------------------------------------


def test_eval_mode_makes_per_image_gradients_independent(images):
    """The assumption the entire refactor rests on.

    d loss / d delta[i] must depend on image i's delta ALONE. If it does not,
    then "which images does this augmentation hurt" is contaminated by the rest
    of the batch, and correlating those numbers across images measures nothing.

    In eval() BatchNorm normalizes with its frozen running statistics, so image
    i's forward pass never sees the other images. Bumping delta[2] must therefore
    move entry 2 of the gradient and nothing else.
    """
    model = _TinySegModel()
    model.eval()
    op = DIFFERENTIABLE_PERTURBATIONS["lighter_R"]

    base = _grad_wrt_delta(model, images, torch.tensor([0.5, 0.5, 0.5, 0.5]), op)
    bumped = _grad_wrt_delta(model, images, torch.tensor([0.5, 0.5, 0.9, 0.5]), op)

    for i in (0, 1, 3):
        assert torch.allclose(base[i], bumped[i], atol=1e-6), (
            f"image {i}'s sensitivity moved when only image 2's delta changed -- "
            f"something is coupling the images"
        )
    assert not torch.isclose(base[2], bumped[2], atol=1e-6)


def test_train_mode_batchnorm_couples_images(images):
    """The counterexample, which is why model.eval() in _sweep is load-bearing
    rather than incidental tidiness.

    In train mode BatchNorm normalizes with BATCH statistics, so image 2's delta
    leaks into every other image's gradient and the independence above collapses.
    If this test ever starts failing, the coupling is gone and the eval() call
    could be revisited -- until then, do not remove it.
    """
    model = _TinySegModel()
    model.train()
    op = DIFFERENTIABLE_PERTURBATIONS["lighter_R"]

    base = _grad_wrt_delta(model, images, torch.tensor([0.5, 0.5, 0.5, 0.5]), op)
    bumped = _grad_wrt_delta(model, images, torch.tensor([0.5, 0.5, 0.9, 0.5]), op)

    assert not torch.allclose(base[0], bumped[0], atol=1e-6)


# --- the probe ---------------------------------------------------------------


def test_grad_for_op_returns_one_value_per_image(images):
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook(interval=1, per_image_delta=True, ref_magnitude=0.3)

    grad = hook._grad_for_op(
        "lighter_R",
        model,
        DIFFERENTIABLE_PERTURBATIONS["lighter_R"],
        images,
        model.data_preprocessor.mean.view(1, -1, 1, 1),
        model.data_preprocessor.std.view(1, -1, 1, 1),
        None,
        np.full(4, 0.3),
    )

    assert isinstance(grad, np.ndarray)
    assert grad.shape == (4,)
    assert np.isfinite(grad).all()
    # The images differ, so their sensitivities should too -- a constant vector
    # would mean the per-image delta collapsed somewhere.
    assert grad.std() > 0


def test_scalar_mode_still_returns_a_single_batch_averaged_value(images):
    """The legacy path stays runnable for A/B comparison."""
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook(interval=1, per_image_delta=False, ref_magnitude=0.3)

    grad = hook._grad_for_op(
        "lighter_R",
        model,
        DIFFERENTIABLE_PERTURBATIONS["lighter_R"],
        images,
        model.data_preprocessor.mean.view(1, -1, 1, 1),
        model.data_preprocessor.std.view(1, -1, 1, 1),
        None,
        np.full(1, 0.3),
    )
    assert grad.shape == (1,)


def test_probe_batch_covers_every_op_with_a_per_image_vector(images):
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook(interval=1)

    grads, _ = hook._probe_batch(model, _batch(images), {})

    assert set(grads) == set(DIFFERENTIABLE_PERTURBATIONS)
    for name, value in grads.items():
        assert value.shape == (4,), f"{name} did not return one value per image"
        assert np.isfinite(value).all(), f"{name} returned a non-finite gradient"


def test_noise_gradient_is_reproducible_across_batches(images):
    """`noise`'s gradient is a projection of the pixel gradient onto its random
    sample. A fresh draw every batch would turn its row into measurement noise
    and attenuate every correlation it takes part in, so the probe pins the RNG:
    two probes of the same batch and model must agree exactly."""
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook(interval=1, probe_seed=7)

    first, _ = hook._probe_batch(model, _batch(images), {})
    second, _ = hook._probe_batch(model, _batch(images), {})

    np.testing.assert_allclose(first["noise"], second["noise"])


# --- the sweep ---------------------------------------------------------------


def test_sweep_visits_every_batch_exactly_once(images, tmp_path):
    """The frozen frame is only worth anything if it is the WHOLE val set. A
    sweep that re-read the first batch, or stopped early, would silently shrink N
    and re-introduce the image-repeat bias the sweep exists to remove."""
    model = _TinySegModel()
    model.eval()
    torch.manual_seed(2)
    batches = [_batch(torch.rand(4, 3, 16, 16)) for _ in range(3)]
    hook = _sweep_hook(tmp_path, batches)

    hook._sweep(_FakeRunner(model, iteration=100), checkpoint=1.0)

    for name in DIFFERENTIABLE_PERTURBATIONS:
        assert len(hook.grad_buffer[name]) == 3, f"{name}: one entry per batch"
        assert all(entry.shape == (4,) for entry in hook.grad_buffer[name])

    # Each buffered entry must be the gradient of ITS OWN batch, not the first
    # batch three times.
    expected = [hook._probe_batch(model, b, {})[0]["lighter_R"] for b in batches]
    for got, want in zip(hook.grad_buffer["lighter_R"], expected):
        np.testing.assert_allclose(got, want)


def test_sweep_keeps_all_op_rows_aligned(images, tmp_path):
    """stack_probe_buffer asserts every op row has the same length. That holds
    only because a batch is probed all-or-nothing across ops."""
    model = _TinySegModel()
    model.eval()
    hook = _sweep_hook(tmp_path, [_batch(images), _batch(images)])

    hook._sweep(_FakeRunner(model), checkpoint=0.5)

    lengths = {
        np.concatenate(hook.grad_buffer[name]).shape[0]
        for name in DIFFERENTIABLE_PERTURBATIONS
    }
    assert lengths == {8}


def test_sweep_leaves_the_model_frozen(images, tmp_path):
    """"Frozen frame" is the entire claim: every column of D_grad must describe
    the SAME weights. torch.autograd.grad populates no parameter .grad and the
    sweep steps no optimizer, so the weights must come out bit-identical."""
    model = _TinySegModel()
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hook = _sweep_hook(tmp_path, [_batch(images), _batch(images)])

    hook._sweep(_FakeRunner(model), checkpoint=0.5)

    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"the sweep moved {key}"
        # BN running stats included: a train-mode forward would have mutated them.
    assert all(p.grad is None for p in model.parameters()), (
        "the sweep populated parameter .grad -- it would pollute the next "
        "optimizer step"
    )
    assert model.training, "the sweep left the model in eval mode"


def test_sweep_restores_rng_state(images, tmp_path):
    """The sweep reseeds the RNG (so `noise` is reproducible). It must put the
    state back, or it silently perturbs the training stream."""
    model = _TinySegModel()
    model.train()
    hook = _sweep_hook(tmp_path, [_batch(images)], probe_seed=1234)

    rng_before = torch.get_rng_state()
    hook._sweep(_FakeRunner(model), checkpoint=1.0)

    assert torch.equal(torch.get_rng_state(), rng_before), (
        "sweep leaked its RNG state into the training stream"
    )


# --- guards ------------------------------------------------------------------


def _exploding_op(images, magnitude):
    return images * (magnitude.reshape(-1, 1, 1, 1) / 0.0)


def test_non_finite_gradient_raises_probe_error(images):
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook(interval=1)

    with pytest.raises(ProbeError):
        hook._grad_for_op(
            "exploding",
            model,
            _exploding_op,
            images,
            model.data_preprocessor.mean.view(1, -1, 1, 1),
            model.data_preprocessor.std.view(1, -1, 1, 1),
            None,
            np.full(4, 0.5),
        )


def test_probe_error_is_not_swallowed(images, tmp_path, monkeypatch):
    """The sweep's blanket `except Exception` exists so a TRANSIENT failure (OOM,
    dataloader hiccup) costs one batch rather than the run. A structurally broken
    gradient is not that, and must not be quietly downgraded into a skipped batch
    -- if it were, every guard in the probe would be decorative.
    """
    model = _TinySegModel()
    monkeypatch.setattr(
        grad_hook, "DIFFERENTIABLE_PERTURBATIONS", {"exploding": _exploding_op}
    )
    hook = _sweep_hook(tmp_path, [_batch(images)], names=["exploding"])

    with pytest.raises(ProbeError):
        hook._sweep(_FakeRunner(model), checkpoint=1.0)


def test_transient_failure_skips_only_that_batch(images, tmp_path, monkeypatch):
    """A flaky batch must cost that batch, not the whole sweep -- losing the
    sweep would lose the entire checkpoint's R."""
    calls = {"n": 0}

    def _flaky(images, magnitude):
        calls["n"] += 1
        if calls["n"] == 1:  # first batch's first op blows up
            raise RuntimeError("CUDA out of memory (simulated)")
        return images * magnitude.reshape(-1, 1, 1, 1)

    model = _TinySegModel()
    monkeypatch.setattr(grad_hook, "DIFFERENTIABLE_PERTURBATIONS", {"flaky": _flaky})
    hook = _sweep_hook(tmp_path, [_batch(images), _batch(images)], names=["flaky"])

    hook._sweep(_FakeRunner(model), checkpoint=1.0)

    assert len(hook.grad_buffer["flaky"]) == 1, (
        "the failed batch should be skipped and the surviving one kept"
    )


# --- interval gating + logging -----------------------------------------------


def test_sweep_fires_on_its_own_clock_not_on_val_epochs(images, tmp_path):
    """The property this pipeline exists for.

    The sweep used to fire from ``after_val_epoch``, which under --aug-type=ours is
    called by RobustValLoop itself: the correlation measurement could only happen on
    an SA round, and for any other aug type it never happened at all. It now runs
    from ``after_train_iter`` on its own interval, so it is reachable from neither
    the val loop nor the SA pipeline.
    """
    model = _TinySegModel()
    model.eval()
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=50)
    runner = _FakeRunner(model, max_iters=100)

    # A val epoch is no longer a trigger, even standing exactly on a firing
    # iteration: only the base Hook's no-op after_val_epoch runs, and nothing sweeps.
    runner.iter = 49
    hook.after_val_epoch(runner)
    assert hook.grad_buffer["lighter_R"] == [], (
        "a val epoch must not drive the correlation pipeline any more"
    )

    # runner.iter is the 0-based index of the iteration that JUST finished, so the
    # 50th iteration is iter=49.
    runner.iter = 20
    hook.after_train_iter(runner, batch_idx=20)
    assert hook.grad_buffer["lighter_R"] == []

    runner.iter = 49  # 50th iteration -- first sweep
    hook.after_train_iter(runner, batch_idx=49)
    assert len(hook.grad_buffer["lighter_R"]) == 1

    runner.iter = 60  # between intervals, no sweep
    hook.after_train_iter(runner, batch_idx=60)
    assert len(hook.grad_buffer["lighter_R"]) == 1

    # The last iteration satisfies BOTH conditions of the gate (100 % 50 == 0, and
    # it is the final iteration). It must sweep once, not twice.
    runner.iter = 99
    hook.after_train_iter(runner, batch_idx=99)
    assert len(hook.grad_buffer["lighter_R"]) == 2


def test_final_iteration_always_sweeps(images, tmp_path):
    """max_iters is rarely a clean multiple of the interval, and the end-of-training
    R -- the converged model's -- is the one anybody actually reads."""
    model = _TinySegModel()
    model.eval()
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=50)
    runner = _FakeRunner(model, max_iters=120)  # 120 % 50 != 0

    runner.iter = 119  # 120th and last iteration
    hook.after_train_iter(runner, batch_idx=119)
    assert len(hook.grad_buffer["lighter_R"]) == 1


def test_a_resumed_run_does_not_re_sweep_passed_intervals(images, tmp_path):
    """The old gate was a `_next_checkpoint` index into a tuple of progress
    fractions. It was reconstructed at 0 on resume, so a run resumed at 60% re-fired
    the 25% and 50% sweeps -- against a model state those checkpoints never saw, and
    overwriting the R they had already produced. A modulo gate holds no state across
    the restart, so there is nothing left to get wrong.
    """
    model = _TinySegModel()
    model.eval()
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=500)
    runner = _FakeRunner(model, max_iters=1000)

    runner.iter = 600  # a fresh hook on a run resumed past the first interval
    hook.after_train_iter(runner, batch_idx=600)
    assert hook.grad_buffer["lighter_R"] == [], (
        "a resumed run must not re-sweep an interval it already passed"
    )

    runner.iter = 999  # the next real firing point
    hook.after_train_iter(runner, batch_idx=999)
    assert len(hook.grad_buffer["lighter_R"]) == 1


def test_sweep_logs_every_batch_per_image(images, tmp_path):
    """A (B,) numpy array is not JSON-serializable -- np.ndarray and np.float32
    both raise TypeError in json.dumps -- and this write happens OUTSIDE the
    probe's try/except, so getting it wrong takes the whole run down rather than
    costing one batch. The log is also what makes R recomputable offline."""
    model = _TinySegModel()
    model.eval()
    hook = _sweep_hook(tmp_path, [_batch(images), _batch(images)])

    hook._sweep(_FakeRunner(model, iteration=40), checkpoint=0.5)

    records = [
        json.loads(line)
        for line in Path(hook.grad_log_path).read_text().strip().splitlines()
    ]
    assert len(records) == 2  # one per sweep batch
    for record in records:
        assert record["checkpoint"] == 0.5
        assert record["iter"] == 40
        assert record["batch_size"] == 4
        assert len(record["grads"]["lighter_R"]) == 4  # per image, not one number
        assert len(record["grads"]) == len(DIFFERENTIABLE_PERTURBATIONS)


# --- SA-driven probe magnitudes ----------------------------------------------
#
# The probe used to hold every op at a hardcoded 0.5, which is not commensurable
# across them (at 0.5, `blur` moves a pixel by at most 78/255 and `noise` by
# 251/255), so R correlated ops held at arbitrary relative strengths. These cover
# the handoff that replaces that constant with the SA-derived distribution.


def _snapshot():
    """What RobustValLoop.publish_corr_magnitudes puts on the runner."""
    return {
        "lighter_R": {"levels": [0.2, 0.7], "probs": [0.8, 0.2]},
        "blur": {"levels": [0.05, 0.9], "probs": [0.1, 0.9]},
    }


def test_defaults_to_the_modal_magnitude(images, tmp_path):
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=1)
    model = _TinySegModel()
    runner = _FakeRunner(model, iteration=0)
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)
    info = runner.aug_grad_magnitude_info

    assert info["source"] == "sa_snapshot"
    assert info["mode"] == "mode"
    assert info["per_op"]["lighter_R"]["mean"] == pytest.approx(0.2)  # p=0.8
    assert info["per_op"]["blur"]["mean"] == pytest.approx(0.9)  # p=0.9
    # An op the snapshot does not cover must still be probed, at the fallback.
    assert info["per_op"]["noise"]["mean"] == pytest.approx(hook.ref_magnitude)


def test_modal_magnitude_is_constant_within_the_sweep(images, tmp_path):
    """min == max is the check that the default mode did not introduce a second
    source of variance into every row of D_grad."""
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=1)
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)

    for name, summary in runner.aug_grad_magnitude_info["per_op"].items():
        assert summary["min"] == summary["max"], name


def test_falls_back_to_fixed_without_a_snapshot(images, tmp_path):
    """Before the first post-warmup SA round, and for a whole --no-corr-sa run,
    there is nothing published. The probe must still run, at ref_magnitude."""
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=1, ref_magnitude=0.5)
    runner = _FakeRunner(_TinySegModel())  # no corr_magnitudes attribute at all

    hook._sweep(runner, checkpoint=0.5)
    info = runner.aug_grad_magnitude_info

    assert info["source"] == "fixed"
    for summary in info["per_op"].values():
        assert summary["mean"] == pytest.approx(0.5)


def test_fixed_mode_ignores_a_live_snapshot(images, tmp_path):
    """The regression guard: an old run must stay reproducible."""
    hook = _sweep_hook(
        tmp_path, [_batch(images)], interval=1, magnitude_mode="fixed", ref_magnitude=0.5
    )
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)

    assert runner.aug_grad_magnitude_info["source"] == "fixed"
    for summary in runner.aug_grad_magnitude_info["per_op"].values():
        assert summary["mean"] == pytest.approx(0.5)


def test_sampled_mode_varies_magnitudes_across_images(images, tmp_path):
    hook = _sweep_hook(
        tmp_path, [_batch(images)], interval=1, magnitude_mode="sampled_shared"
    )
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)

    covered = runner.aug_grad_magnitude_info["per_op"]["lighter_R"]
    assert covered["min"] < covered["max"], "sampled mode collapsed to a constant"


def test_sampled_magnitudes_advance_across_batches(images, tmp_path):
    """The magnitude RNG is seeded once per sweep, not per batch. Reseeding per
    batch would hand every batch the same draw -- and at sweep_batch_size=1 that
    collapses the whole val set onto a single magnitude."""
    batches = [_batch(torch.rand(1, 3, 16, 16)) for _ in range(6)]
    hook = _sweep_hook(tmp_path, batches, interval=1, magnitude_mode="sampled_shared")
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)

    summary = runner.aug_grad_magnitude_info["per_op"]["lighter_R"]
    assert summary["min"] < summary["max"]


def test_snapshot_is_read_once_per_sweep(images, tmp_path):
    """A snapshot arriving mid-sweep must not split one R across two magnitude
    regimes: every column of the matrix has to describe one magnitude setting as
    well as one model state."""
    batches = [_batch(torch.rand(1, 3, 16, 16)) for _ in range(4)]
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    class _MutatingList(list):
        def __iter__(self):
            for i, item in enumerate(list.__iter__(self)):
                if i == 2:  # SA lands halfway through
                    runner.corr_magnitudes = {
                        "lighter_R": {"levels": [0.99], "probs": [1.0]}
                    }
                yield item

    hook = _sweep_hook(tmp_path, _MutatingList(batches), interval=1)
    hook._sweep(runner, checkpoint=0.5)

    summary = runner.aug_grad_magnitude_info["per_op"]["lighter_R"]
    assert summary["min"] == summary["max"] == pytest.approx(0.2)


def test_magnitudes_are_logged_with_the_gradients(images, tmp_path):
    """A gradient is only interpretable together with the magnitude it was taken
    at; without them an offline recompute cannot tell a fixed-0.5 sweep from an
    SA-driven one."""
    hook = _sweep_hook(tmp_path, [_batch(images)], interval=1)
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)

    record = json.loads(Path(hook.grad_log_path).read_text().splitlines()[0])
    assert record["magnitude_source"] == "sa_snapshot"
    assert record["magnitude_mode"] == "mode"
    assert set(record["magnitudes"]) == set(DIFFERENTIABLE_PERTURBATIONS)
    assert record["magnitudes"]["lighter_R"] == [0.2] * 4


def test_rejects_an_unknown_magnitude_mode():
    with pytest.raises(ValueError, match="unknown magnitude_mode"):
        CollectGradientHook(interval=1, magnitude_mode="argmax")


def test_seeded_snapshot_pins_a_run_with_no_sa(images, tmp_path):
    """--corr-magnitudes: how a --no-corr-sa control arm gets measured at the same
    magnitudes as the SA-on run it is compared against."""
    path = tmp_path / "corr_magnitudes.json"
    path.write_text(json.dumps([{"iter": 10, "magnitudes": {
        "lighter_R": {"levels": [0.35], "probs": [1.0], "mode": 0.35}}}]))

    hook = _sweep_hook(
        tmp_path, [_batch(images)], interval=1, magnitudes_path=str(path)
    )
    hook._seed_snapshot = hook._load_seed_snapshot()
    runner = _FakeRunner(_TinySegModel())  # no live SA

    hook._sweep(runner, checkpoint=0.5)
    info = runner.aug_grad_magnitude_info

    assert info["source"] == "seeded_snapshot"
    assert info["per_op"]["lighter_R"]["mean"] == pytest.approx(0.35)


def test_a_live_snapshot_supersedes_a_seeded_one(images, tmp_path):
    path = tmp_path / "corr_magnitudes.json"
    path.write_text(json.dumps([{"iter": 10, "magnitudes": {
        "lighter_R": {"levels": [0.35], "probs": [1.0]}}}]))

    hook = _sweep_hook(
        tmp_path, [_batch(images)], interval=1, magnitudes_path=str(path)
    )
    hook._seed_snapshot = hook._load_seed_snapshot()
    runner = _FakeRunner(_TinySegModel())
    runner.corr_magnitudes = _snapshot()

    hook._sweep(runner, checkpoint=0.5)
    info = runner.aug_grad_magnitude_info

    assert info["source"] == "sa_snapshot"
    assert info["per_op"]["lighter_R"]["mean"] == pytest.approx(0.2)


def test_a_malformed_seed_file_raises_rather_than_degrading(tmp_path):
    """Silently probing at 0.5 when a pinned magnitude was requested produces a
    control that is comparable to nothing, and nothing downstream reveals it."""
    path = tmp_path / "corr_magnitudes.json"
    path.write_text("[]")
    hook = CollectGradientHook(interval=1, magnitudes_path=str(path))
    with pytest.raises(ValueError, match="no snapshots"):
        hook._load_seed_snapshot()
