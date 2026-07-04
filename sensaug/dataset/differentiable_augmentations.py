"""Differentiable (autograd-compatible), GPU-batched image perturbation ops.

Ported from azshue/adversarial_data_augmentation's data/diffaug.py (Shu, Shen,
Lin, Goldstein -- "Adversarial Differentiable Data Augmentation for Autonomous
Systems", ICRA 2021). This module contains ONLY the 8 atomic perturbation ops
(+ a name-keyed registry over the existing sensaug 14-key perturbation
vocabulary). It does NOT contain the PGD/FGSM adversarial magnitude-search
loop (see the reference repo's train.py) -- that is future, out-of-scope work.

Tensor contract (all public functions):
    - dtype: torch.float32
    - shape: (B, 3, H, W)          <- BCHW, batched
    - value range: [0, 1]
    - channel order: RGB (0=R, 1=G, 2=B)

This is intentionally a DIFFERENT convention from sensaug.dataset.augmentations
"""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import kornia.color

Tensor = torch.Tensor

# --- channel index constants -------------------------------------------------
R, G, B = 0, 1, 2  # index into the RGB channel dim
H, S, V = 0, 1, 2  # index into kornia's HSV channel dim

RGB_RAIL_MAX = 1.0
# kornia.color.rgb_to_hsv returns hue in RADIANS [0, 2*pi), NOT normalized
# [0, 1] like sensaug.dataset.augmentations.HSVPerturbation's cv2-based hue.
_HSV_RAIL_MAX = {H: 2 * math.pi, S: 1.0, V: 1.0}
# Matches augmentations.py's HSVPerturbation: darkening V rails toward a
# nonzero floor instead of pure black.
V_DARKEN_FLOOR = 10.0 / 255.0

__all__ = [
    "color_channel",
    "hsv_channel",
    "blur",
    "gaussian_noise",
    "DiffAugment",
    "DIFFERENTIABLE_PERTURBATIONS",
    "R",
    "G",
    "B",
    "H",
    "S",
    "V",
]


def _as_delta(delta: Union[float, Tensor], reference: Tensor) -> Tensor:
    """Normalize a python float or a (possibly requires_grad) tensor onto
    `reference`'s dtype/device, preserving the autograd graph if `delta` is
    already a tensor."""
    if torch.is_tensor(delta):
        return delta.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(float(delta), dtype=reference.dtype, device=reference.device)


def _weighted_rail_perturb(
    channel: Tensor, delta: Tensor, rail_min: float, rail_max: float
) -> Tensor:
    """out = channel * (1 - |delta|) + rail * |delta|, rail = rail_max if
    delta >= 0 else rail_min. Same weighted-average-toward-a-rail-value
    formula as sensaug.dataset.augmentations.perturb_rgb / HSVPerturbation --
    kept for numeric continuity with the rest of the repo instead of the
    reference repo's multiplicative `*= (1 + delta)`. Differentiable in
    `channel` and in the magnitude of `delta`; the SIGN of delta only selects
    which constant rail is used (a hard, non-differentiable branch), matching
    the existing repo convention.
    """
    rail = rail_max if float(delta.detach()) >= 0 else rail_min
    abs_delta = delta.abs()
    return channel * (1.0 - abs_delta) + rail * abs_delta


def _replace_channel(
    tensor: Tensor, channel: int, num_channels: int, new_value: Tensor
) -> Tensor:
    """Out-of-place equivalent of `tensor[:, channel] = new_value`, built via
    torch.cat instead of in-place indexed assignment. Reference repo's ops use
    torch.stack the same way -- deliberately avoided in-place writes here too,
    since `tensor[:, channel] = f(tensor[:, channel])` corrupts the autograd
    graph when `tensor` is itself the output of a differentiable op whose
    backward needs to read that same slice (bit us in practice on
    kornia.color.rgb_to_hsv/hsv_to_rgb: the read is saved for backward, then
    the in-place write invalidates it via PyTorch's version counter)."""
    parts = [
        new_value if c == channel else tensor[:, c : c + 1] for c in range(num_channels)
    ]
    return torch.cat(parts, dim=1)


def color_channel(images: Tensor, channel: int, delta: Union[float, Tensor]) -> Tensor:
    """Push one RGB channel toward 0 (delta<0) or 1 (delta>0). Signed delta
    directly encodes direction (reference repo's color_R/color_G/color_B)."""
    assert channel in (R, G, B), "channel must be R, G, or B (0, 1, 2)"
    delta_t = _as_delta(delta, images)
    new_value = _weighted_rail_perturb(
        images[:, channel : channel + 1], delta_t, rail_min=0.0, rail_max=RGB_RAIL_MAX
    )
    return _replace_channel(images, channel, 3, new_value)


def hsv_channel(images: Tensor, channel: int, delta: Union[float, Tensor]) -> Tensor:
    """Push one HSV channel toward its rail; converts RGB->HSV->RGB via
    kornia. Signed delta directly encodes direction (reference repo's
    color_H/color_S/color_V)."""
    assert channel in (H, S, V), "channel must be H, S, or V (0, 1, 2)"
    delta_t = _as_delta(delta, images)
    hsv = kornia.color.rgb_to_hsv(images)
    rail_max = _HSV_RAIL_MAX[channel]
    rail_min = 0.0
    if channel == V and float(delta_t.detach()) < 0:
        rail_min = V_DARKEN_FLOOR
    new_value = _weighted_rail_perturb(
        hsv[:, channel : channel + 1], delta_t, rail_min=rail_min, rail_max=rail_max
    )
    hsv_out = _replace_channel(hsv, channel, 3, new_value)
    return kornia.color.hsv_to_rgb(hsv_out).clamp(0.0, 1.0)


def _gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tensor) -> Tensor:
    """Hand-built (guaranteed differentiable w.r.t. `sigma`) 2D Gaussian
    kernel, shape (1, 1, kh, kw). Mirrors the reference repo's own "custom
    conv2d" blur -- deliberately not kornia.filters.gaussian_blur2d, whose
    sigma-argument autograd support is version-dependent."""
    kh, kw = kernel_size
    ax_y = torch.arange(kh, dtype=sigma.dtype, device=sigma.device) - (kh - 1) / 2.0
    ax_x = torch.arange(kw, dtype=sigma.dtype, device=sigma.device) - (kw - 1) / 2.0
    var = 2 * sigma**2 + 1e-12
    kernel2d = torch.outer(torch.exp(-(ax_y**2) / var), torch.exp(-(ax_x**2) / var))
    return (kernel2d / kernel2d.sum()).view(1, 1, kh, kw)


def blur(
    images: Tensor, sigma: Union[float, Tensor], kernel_size: Tuple[int, int] = (67, 67)
) -> Tensor:
    """Gaussian blur; magnitude == sigma directly (unsigned; no lighter/darker
    direction, param_min = 0.0). kernel_size is a fixed hyperparameter, not
    swept -- matches the reference repo. NOTE: this is NOT numerically
    equivalent to augmentations.py's Blur/GaussianBlurPerturbation, which
    instead derive kernel_size from magnitude via cv2's implicit sigma."""
    sigma_t = _as_delta(sigma, images).clamp(min=1e-3)
    kernel = _gaussian_kernel2d(kernel_size, sigma_t).to(images.dtype)
    kernel = kernel.repeat(images.shape[1], 1, 1, 1)  # depthwise: one copy per channel
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    return F.conv2d(images, kernel, padding=padding, groups=images.shape[1])


def gaussian_noise(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Additive Gaussian noise scaled by delta (signed; param_min = -eps)."""
    delta_t = _as_delta(delta, images)
    noise = torch.randn(images.shape, dtype=images.dtype, device=images.device)
    return torch.clamp(images + noise * delta_t, 0.0, 1.0)


class DiffAugment:
    """GPU-batched, autograd-compatible perturbation dispatcher, structurally
    mirroring azshue/adversarial_data_augmentation's diffaug.py DiffAugment
    class (same aug_id scheme, same single_aug(img, aug_id, delta) ->
    (perturbed_img, param_min) contract) so a future PGD/FGSM magnitude-search
    loop (this repo's analog of the reference's train.py -- out of scope
    here) can plug in directly."""

    AUG_IDS = ("1", "2", "R", "G", "B", "H", "S", "V")
    _RGB_CHANNEL = {"R": R, "G": G, "B": B}
    _HSV_CHANNEL = {"H": H, "S": S, "V": V}

    def __init__(self, eps: float = 1.0, kernel_size: Tuple[int, int] = (67, 67)):
        self.eps = eps
        self.kernel_size = kernel_size

    def param_min(self, aug_id: str) -> float:
        return 0.0 if aug_id == "1" else -self.eps

    def single_aug(
        self, img: Tensor, aug_id: str, delta: Union[float, Tensor]
    ) -> Tuple[Tensor, float]:
        if aug_id == "1":
            out = blur(img, sigma=delta, kernel_size=self.kernel_size)
        elif aug_id == "2":
            out = gaussian_noise(img, delta=delta)
        elif aug_id in self._RGB_CHANNEL:
            out = color_channel(img, self._RGB_CHANNEL[aug_id], delta)
        elif aug_id in self._HSV_CHANNEL:
            out = hsv_channel(img, self._HSV_CHANNEL[aug_id], delta)
        else:
            raise ValueError(f"Unknown aug_id: {aug_id!r}, expected one of {self.AUG_IDS}")
        return out, self.param_min(aug_id)


def _named_rgb_op(channel: int, sign: float):
    def _op(images: Tensor, magnitude: Union[float, Tensor]) -> Tensor:
        return color_channel(images, channel, sign * _as_delta(magnitude, images))

    return _op


def _named_hsv_op(channel: int, sign: float):
    def _op(images: Tensor, magnitude: Union[float, Tensor]) -> Tensor:
        return hsv_channel(images, channel, sign * _as_delta(magnitude, images))

    return _op


# Name-keyed registry matching the 14-key vocabulary already used by the
# legacy augmentations.PERTURBATIONS / gpr_sa.PERTURBATIONS / bopt_sa.PERTURBATIONS
# / sensitivity_analysis.py's non-"_new" path. Each value is a callable
# (images, magnitude) -> perturbed_images, unsigned magnitude, direction baked
# into the key -- mirroring augmentations.py's LighterR/DarkerR/... classes.
# NOT numerically calibrated against those classes' magnitude ranges (see
# module docstring); this registry exists so the vocabulary is
# recognizable/pluggable, not to guarantee identical scale.
DIFFERENTIABLE_PERTURBATIONS = {
    "lighter_R": _named_rgb_op(R, +1),
    "darker_R": _named_rgb_op(R, -1),
    "lighter_G": _named_rgb_op(G, +1),
    "darker_G": _named_rgb_op(G, -1),
    "lighter_B": _named_rgb_op(B, +1),
    "darker_B": _named_rgb_op(B, -1),
    "lighter_H": _named_hsv_op(H, +1),
    "darker_H": _named_hsv_op(H, -1),
    "lighter_S": _named_hsv_op(S, +1),
    "darker_S": _named_hsv_op(S, -1),
    "lighter_V": _named_hsv_op(V, +1),
    "darker_V": _named_hsv_op(V, -1),
    "blur": lambda images, magnitude: blur(images, sigma=magnitude),
    "noise": lambda images, magnitude: gaussian_noise(images, delta=magnitude),
}
