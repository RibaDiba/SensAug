"""Tests for CollectGradientHook's per-image gradient probe.

Requires the full mmseg/mmengine stack (grad_hook imports the registries), so run
these in the `sensaug` conda env on a compute node, not on a laptop.
"""

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.dataset.differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS
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
    def __init__(self, model, iteration=0):
        self.model = model
        self.iter = iteration
        self.val_loop = types.SimpleNamespace(sa_curve={"placeholder": [1]})


@pytest.fixture
def images():
    torch.manual_seed(1)
    return torch.rand(4, 3, 16, 16)


def _probe_hook(model, images, **kwargs):
    hook = CollectGradientHook(**kwargs)
    hook.grad_buffer = {name: [] for name in DIFFERENTIABLE_PERTURBATIONS}
    hook._next_probe_batch = lambda: {"inputs": images, "data_samples": None}
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
    """The counterexample, which is why model.eval() in _probe_gradients is
    load-bearing rather than incidental tidiness.

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
    hook = CollectGradientHook(per_image_delta=True, ref_magnitude=0.3)

    grad = hook._grad_for_op(
        "lighter_R",
        model,
        DIFFERENTIABLE_PERTURBATIONS["lighter_R"],
        images,
        model.data_preprocessor.mean.view(1, -1, 1, 1),
        model.data_preprocessor.std.view(1, -1, 1, 1),
        None,
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
    hook = CollectGradientHook(per_image_delta=False, ref_magnitude=0.3)

    grad = hook._grad_for_op(
        "lighter_R",
        model,
        DIFFERENTIABLE_PERTURBATIONS["lighter_R"],
        images,
        model.data_preprocessor.mean.view(1, -1, 1, 1),
        model.data_preprocessor.std.view(1, -1, 1, 1),
        None,
    )
    assert grad.shape == (1,)


def test_probe_covers_every_op_with_a_per_image_vector(images):
    model = _TinySegModel()
    model.eval()
    hook = _probe_hook(model, images)

    grads = hook._probe_gradients(_FakeRunner(model))

    assert set(grads) == set(DIFFERENTIABLE_PERTURBATIONS)
    for name, value in grads.items():
        assert value.shape == (4,), f"{name} did not return one value per image"
        assert np.isfinite(value).all(), f"{name} returned a non-finite gradient"


# --- guards ------------------------------------------------------------------


def _exploding_op(images, magnitude):
    return images * (magnitude.reshape(-1, 1, 1, 1) / 0.0)


def test_non_finite_gradient_raises_probe_error(images):
    model = _TinySegModel()
    model.eval()
    hook = CollectGradientHook()

    with pytest.raises(ProbeError):
        hook._grad_for_op(
            "exploding",
            model,
            _exploding_op,
            images,
            model.data_preprocessor.mean.view(1, -1, 1, 1),
            model.data_preprocessor.std.view(1, -1, 1, 1),
            None,
        )


def test_probe_error_is_not_swallowed(images, monkeypatch):
    """_probe_gradients' blanket `except Exception` exists so a TRANSIENT failure
    (OOM, dataloader hiccup) costs one probe rather than the run. A structurally
    broken gradient is not that, and must not be quietly downgraded into a skipped
    probe -- if it were, every guard in the probe would be decorative.
    """
    model = _TinySegModel()
    hook = _probe_hook(model, images)
    monkeypatch.setattr(
        grad_hook, "DIFFERENTIABLE_PERTURBATIONS", {"exploding": _exploding_op}
    )

    with pytest.raises(ProbeError):
        hook._probe_gradients(_FakeRunner(model))


def test_transient_failure_is_swallowed_into_a_skipped_probe(images, monkeypatch):
    def _oom(images, magnitude):
        raise RuntimeError("CUDA out of memory (simulated)")

    model = _TinySegModel()
    hook = _probe_hook(model, images)
    monkeypatch.setattr(grad_hook, "DIFFERENTIABLE_PERTURBATIONS", {"flaky": _oom})

    assert hook._probe_gradients(_FakeRunner(model)) == {}


def test_probe_restores_rng_state_and_training_mode(images):
    """The probe reseeds the RNG (so `noise` is reproducible) and flips the model
    to eval. Both must be put back, or the probe silently perturbs training."""
    model = _TinySegModel()
    model.train()
    hook = _probe_hook(model, images, probe_seed=1234)

    rng_before = torch.get_rng_state()
    grads = hook._probe_gradients(_FakeRunner(model))

    assert grads, "probe should have succeeded"
    assert torch.equal(torch.get_rng_state(), rng_before), (
        "probe leaked its RNG state into the training stream"
    )
    assert model.training, "probe left the model in eval mode"


def test_noise_gradient_is_reproducible_across_probes(images):
    """`noise`'s gradient is a projection of the pixel gradient onto its random
    sample. A fresh draw every probe would turn its row into measurement noise
    and attenuate every correlation it takes part in, so the probe pins the RNG:
    two probes of the same batch and model must agree exactly."""
    model = _TinySegModel()
    model.eval()
    hook = _probe_hook(model, images, probe_seed=7)
    runner = _FakeRunner(model)

    first = hook._probe_gradients(runner)
    second = hook._probe_gradients(runner)

    np.testing.assert_allclose(first["noise"], second["noise"])


# --- logging -----------------------------------------------------------------


def test_after_train_iter_buffers_and_logs_per_image_grads(images, tmp_path):
    """A (B,) numpy array is not JSON-serializable -- np.ndarray and np.float32
    both raise TypeError in json.dumps -- and this write happens in
    after_train_iter, OUTSIDE the probe's try/except, so getting it wrong takes
    the whole training run down rather than costing one probe."""
    model = _TinySegModel()
    model.eval()
    runner = _FakeRunner(model, iteration=50)
    hook = _probe_hook(model, images, probe_interval=50, log_interval=50)
    hook.grad_log_path = str(tmp_path / "aug_gradient_log.txt")

    hook.after_train_iter(runner, batch_idx=0)

    # buffered per probe, as a (B,) array -- probe boundaries stay intact for the
    # cluster bootstrap downstream
    assert len(hook.grad_buffer["lighter_R"]) == 1
    assert hook.grad_buffer["lighter_R"][0].shape == (4,)

    record = json.loads(Path(hook.grad_log_path).read_text().strip())
    assert record["iter"] == 50
    assert record["batch_size"] == 4
    assert len(record["grads"]["lighter_R"]) == 4  # per image, not one number
    assert len(record["grads"]) == len(DIFFERENTIABLE_PERTURBATIONS)
