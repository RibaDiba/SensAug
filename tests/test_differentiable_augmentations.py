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
