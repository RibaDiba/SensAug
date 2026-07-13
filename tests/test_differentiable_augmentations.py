import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.dataset.differentiable_augmentations import (
    DIFFERENTIABLE_PERTURBATIONS,
    DiffAugment,
    R,
    V,
    _broadcast_delta,
    blur,
    color_channel,
    hsv_channel,
)

LEGACY_PERTURBATION_KEYS = {
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
}


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 3, 16, 16)


@pytest.fixture
def big_batch():
    """Batch of 4 with per-image deltas that differ in BOTH magnitude and sign --
    a uniform delta would hide exactly the bugs these tests exist to catch."""
    torch.manual_seed(0)
    return torch.rand(4, 3, 16, 16)


@pytest.mark.parametrize("aug_id", DiffAugment.AUG_IDS)
def test_single_aug_shape_and_finite(batch, aug_id):
    aug = DiffAugment(kernel_size=(15, 15))
    out, _param_min = aug.single_aug(batch, aug_id, delta=0.3)
    assert out.shape == batch.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("aug_id", DiffAugment.AUG_IDS)
def test_single_aug_gradient_flows_to_delta(batch, aug_id):
    """Core correctness requirement: delta must receive a nonzero gradient,
    since this module exists to support a future gradient-based (adversarial)
    magnitude search over delta."""
    aug = DiffAugment(kernel_size=(15, 15))
    param_min = aug.param_min(aug_id)
    # blur's param_min is 0.0 (unsigned magnitude); everything else is signed.
    test_delta = 0.4 if param_min == 0.0 else -0.4
    delta = torch.tensor(test_delta, requires_grad=True)

    out, _ = aug.single_aug(batch, aug_id, delta)
    out.sum().backward()

    assert delta.grad is not None
    assert delta.grad.item() != 0.0


def test_registry_covers_legacy_perturbation_vocabulary():
    """DIFFERENTIABLE_PERTURBATIONS should expose the same 14 string keys as
    sensaug.dataset.augmentations.PERTURBATIONS / gpr_sa.PERTURBATIONS /
    bopt_sa.PERTURBATIONS, without importing those modules (they pull in the
    full mmcv/mmseg/cv2 stack, which this lightweight test suite avoids)."""
    assert set(DIFFERENTIABLE_PERTURBATIONS) == LEGACY_PERTURBATION_KEYS


@pytest.mark.parametrize("name", sorted(LEGACY_PERTURBATION_KEYS))
def test_registry_op_shape_and_finite(batch, name):
    out = DIFFERENTIABLE_PERTURBATIONS[name](batch, 0.3)
    assert out.shape == batch.shape
    assert torch.isfinite(out).all()


def test_color_channel_matches_hand_derived_cpu_formula(batch):
    """sensaug.dataset.augmentations.perturb_rgb (BGR/uint8/[0,255]) computes
    out = channel - alpha*channel + 255*direction*alpha. On a [0,1]-scale
    channel this is out = channel*(1-alpha) + 1*direction*alpha -- verify
    color_channel reproduces it exactly for both directions."""
    magnitude = 0.4

    expected_lighter = batch[:, R] * (1 - magnitude) + 1 * 1 * magnitude
    actual_lighter = color_channel(batch, R, delta=magnitude)[:, R]
    assert torch.allclose(expected_lighter, actual_lighter)

    expected_darker = batch[:, R] * (1 - magnitude) + 1 * 0 * magnitude
    actual_darker = color_channel(batch, R, delta=-magnitude)[:, R]
    assert torch.allclose(expected_darker, actual_darker)


def test_hsv_channel_round_trip_is_identity_at_zero_delta(batch):
    out = hsv_channel(batch, V, delta=0.0)
    assert torch.allclose(out, batch, atol=1e-5)


def test_hsv_darken_v_rails_to_floor_not_black(batch):
    """augmentations.py's HSVPerturbation (and the pre-existing bp_gpu.py
    hsv_gpu_perturb) never crush V to pure black when darkening -- they rail
    toward 10/255 instead. Verify the same floor here."""
    import kornia.color

    out = hsv_channel(batch, V, delta=-1.0)
    v_channel = kornia.color.rgb_to_hsv(out)[:, 2]
    assert torch.allclose(v_channel, torch.full_like(v_channel, 10.0 / 255.0), atol=1e-4)


def test_color_channel_rejects_invalid_channel(batch):
    with pytest.raises(AssertionError):
        color_channel(batch, channel=99, delta=0.3)


def test_diff_augment_unknown_aug_id_raises(batch):
    with pytest.raises(ValueError):
        DiffAugment().single_aug(batch, "not_a_real_id", delta=0.3)


# --- per-image delta ---------------------------------------------------------
#
# Every op must accept a length-B delta (one magnitude per image), because that
# is what lets a single autograd pass measure d loss / d magnitude separately for
# each image -- the measurement the whole redundancy matrix is built on. The
# scalar path must survive untouched.
#
# `noise` is excluded from the equivalence tests below: it draws a fresh sample
# per call, so a batched call and a per-image loop cannot see the same noise. It
# gets its own test.
DETERMINISTIC_KEYS = sorted(LEGACY_PERTURBATION_KEYS - {"noise"})


def test_broadcast_delta_accepts_scalar_and_per_image():
    assert _broadcast_delta(torch.tensor(0.5), batch_size=4).shape == (1, 1, 1, 1)
    assert _broadcast_delta(torch.zeros(4), batch_size=4).shape == (4, 1, 1, 1)


def test_broadcast_delta_rejects_wrong_length():
    with pytest.raises(AssertionError):
        _broadcast_delta(torch.zeros(3), batch_size=4)


@pytest.mark.parametrize("name", DETERMINISTIC_KEYS)
def test_scalar_delta_equals_uniform_per_image_delta(big_batch, name):
    """Back-compat: a uniform per-image delta is the same perturbation as the old
    scalar, just expressed the new way."""
    op = DIFFERENTIABLE_PERTURBATIONS[name]
    scalar = op(big_batch, 0.35)
    vector = op(big_batch, torch.full((4,), 0.35))
    assert torch.allclose(scalar, vector, atol=1e-6)


@pytest.mark.parametrize("name", DETERMINISTIC_KEYS)
def test_per_image_delta_matches_per_image_scalar_loop(big_batch, name):
    """THE golden test: applying a length-B delta in one batched call must equal
    looping over the images and applying each one's delta through the scalar
    path."""
    op = DIFFERENTIABLE_PERTURBATIONS[name]
    deltas = torch.tensor([0.15, 0.62, 0.31, 0.48])

    vectorized = op(big_batch, deltas)
    looped = torch.cat(
        [op(big_batch[i : i + 1], deltas[i]) for i in range(big_batch.shape[0])]
    )
    assert torch.allclose(vectorized, looped, atol=1e-5)


@pytest.mark.parametrize("name", DETERMINISTIC_KEYS)
def test_op_does_not_couple_images(big_batch, name):
    """Image i's output must depend on delta[i] alone. (The model-level version of
    this property, which additionally requires BatchNorm to be frozen, is in
    tests/test_grad_hook.py.)"""
    op = DIFFERENTIABLE_PERTURBATIONS[name]
    base = op(big_batch, torch.tensor([0.3, 0.3, 0.3, 0.3]))
    bumped = op(big_batch, torch.tensor([0.3, 0.3, 0.7, 0.3]))

    for i in (0, 1, 3):
        assert torch.allclose(base[i], bumped[i], atol=1e-6), f"image {i} moved"
    assert not torch.allclose(base[2], bumped[2], atol=1e-6)


@pytest.mark.parametrize("name", sorted(LEGACY_PERTURBATION_KEYS))
def test_per_image_delta_yields_one_gradient_per_image(big_batch, name):
    """A length-B delta must collect a length-B gradient. Each image is weighted
    differently in the loss, so an op that collapsed the delta internally (e.g.
    via delta.mean()) would hand back a uniform gradient instead."""
    op = DIFFERENTIABLE_PERTURBATIONS[name]
    delta = torch.full((4,), 0.4, requires_grad=True)

    out = op(big_batch, delta)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(-1, 1, 1, 1)
    (out * weights).sum().backward()

    assert delta.grad is not None
    assert delta.grad.shape == (4,)
    assert torch.isfinite(delta.grad).all()
    assert (delta.grad != 0).all()


def test_rail_selection_is_per_image_not_per_batch(big_batch):
    """The SIGN of delta picks which rail (0 or 1) the channel is pulled toward.
    With a per-image delta the sign differs across the batch, so a single
    batch-wide `if delta >= 0` would pull half the images toward the wrong rail."""
    deltas = torch.tensor([0.5, -0.5, 0.5, -0.5])
    out = color_channel(big_batch, R, deltas)[:, R]

    for i, d in enumerate(deltas):
        rail = 1.0 if d >= 0 else 0.0
        expected = big_batch[i, R] * (1 - d.abs()) + rail * d.abs()
        assert torch.allclose(out[i], expected, atol=1e-6), f"image {i} wrong rail"


def test_hsv_v_darken_floor_is_selected_per_image(big_batch):
    """Darkening V rails toward 10/255, lightening toward 1.0. With mixed signs in
    one batch, both rails have to be in play simultaneously."""
    import kornia.color

    deltas = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    v_out = kornia.color.rgb_to_hsv(hsv_channel(big_batch, V, deltas))[:, 2]

    floor = 10.0 / 255.0
    for i in (0, 2):  # darkened -> floor, not black
        assert torch.allclose(v_out[i], torch.full_like(v_out[i], floor), atol=1e-4)
    for i in (1, 3):  # lightened -> 1.0
        assert torch.allclose(v_out[i], torch.ones_like(v_out[i]), atol=1e-4)


def test_blur_applies_each_images_own_sigma(big_batch):
    """The per-image blur folds the batch into the channel dim and convolves with
    groups=B*C. The kernels must be repeat_interleave'd, not repeat'd (tiled): a
    tiled stack would blur image b with a DIFFERENT image's sigma, with identical
    shapes and no error -- silently wrong numbers. Deliberately far-apart sigmas,
    so a misassignment cannot hide inside the tolerance."""
    sigmas = torch.tensor([0.5, 3.0, 1.0, 8.0])
    out = blur(big_batch, sigmas, kernel_size=(15, 15))

    for i, sigma in enumerate(sigmas):
        expected = blur(big_batch[i : i + 1], sigma, kernel_size=(15, 15))
        assert torch.allclose(out[i : i + 1], expected, atol=1e-5), (
            f"image {i} was not blurred with its own sigma {float(sigma)}"
        )


def test_noise_applies_each_images_own_delta(big_batch, monkeypatch):
    """noise draws a fresh sample per call, so it cannot be loop-compared. Pin the
    broadcast directly instead: with the sample held fixed, image i must be scaled
    by delta[i]. delta[0] = 0 also pins that a zero magnitude is a no-op."""
    fixed = torch.randn_like(big_batch)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: fixed)

    deltas = torch.tensor([0.0, 0.1, -0.2, 0.4])
    out = DIFFERENTIABLE_PERTURBATIONS["noise"](big_batch, deltas)

    expected = torch.clamp(big_batch + fixed * deltas.view(-1, 1, 1, 1), 0.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[0], big_batch[0], atol=1e-6)
