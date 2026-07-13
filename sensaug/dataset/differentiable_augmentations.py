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

Magnitude contract: every op takes `delta`/`sigma` as EITHER a scalar (one
magnitude for the whole batch) OR a length-B tensor (one magnitude per image).
The per-image form exists so a single autograd pass can measure
d loss / d magnitude separately for each image (sensaug.hooks.grad_hook); the
scalar form is unchanged and still exercised by DiffAugment.

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


def _broadcast_delta(delta: Tensor, batch_size: int) -> Tensor:
    """Reshape a scalar (0-dim) or per-image (B,) delta to (n, 1, 1, 1), n in
    {1, B}, so it broadcasts against a (B, C, H, W) batch.

    The per-image form is what lets a single probe measure d loss / d delta
    SEPARATELY for each image (see sensaug.hooks.grad_hook): autograd then
    returns a length-B gradient instead of one batch-averaged number. A 0-dim
    delta reshapes to (1, 1, 1, 1) and broadcasts identically to before, so the
    scalar path is numerically unchanged.
    """
    d = delta.reshape(-1, 1, 1, 1)
    assert d.shape[0] in (1, batch_size), (
        f"delta must be a scalar or a length-{batch_size} vector (one per "
        f"image), got shape {tuple(delta.shape)}"
    )
    return d


def _as_rail(rail: Union[float, Tensor], reference: Tensor) -> Tensor:
    """Rails are constants (no autograd) but may be per-image tensors -- see
    hsv_channel's V floor."""
    if torch.is_tensor(rail):
        return rail.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(rail, dtype=reference.dtype, device=reference.device)


def _weighted_rail_perturb(
    channel: Tensor,
    delta: Tensor,
    rail_min: Union[float, Tensor],
    rail_max: Union[float, Tensor],
) -> Tensor:
    """out = channel * (1 - |delta|) + rail * |delta|, rail = rail_max if
    delta >= 0 else rail_min. Same weighted-average-toward-a-rail-value
    formula as sensaug.dataset.augmentations.perturb_rgb / HSVPerturbation --
    kept for numeric continuity with the rest of the repo instead of the
    reference repo's multiplicative `*= (1 + delta)`. Differentiable in
    `channel` and in the magnitude of `delta`; the SIGN of delta only selects
    which constant rail is used (a hard, non-differentiable branch), matching
    the existing repo convention.

    `delta` may be scalar or per-image, so the rail is selected elementwise:
    with a per-image delta the sign can differ across the batch.
    """
    d = _broadcast_delta(delta, channel.shape[0])
    rail = torch.where(d >= 0, _as_rail(rail_max, d), _as_rail(rail_min, d))
    abs_delta = d.abs()
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
    rail_min: Union[float, Tensor] = 0.0
    if channel == V:
        # Darkening V rails toward a nonzero floor, not pure black. With a
        # per-image delta the sign varies across the batch, so this has to be an
        # elementwise select rather than a python `if` on the sign.
        d = _broadcast_delta(delta_t, images.shape[0])
        rail_min = torch.where(
            d < 0, torch.full_like(d, V_DARKEN_FLOOR), torch.zeros_like(d)
        )
    new_value = _weighted_rail_perturb(
        hsv[:, channel : channel + 1], delta_t, rail_min=rail_min, rail_max=rail_max
    )
    hsv_out = _replace_channel(hsv, channel, 3, new_value)
    return kornia.color.hsv_to_rgb(hsv_out).clamp(0.0, 1.0)


def _gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tensor) -> Tensor:
    """Hand-built (guaranteed differentiable w.r.t. `sigma`) 2D Gaussian kernel
    STACK, shape (n, 1, kh, kw) with n = 1 for a scalar sigma and n = B for a
    per-image sigma. Mirrors the reference repo's own "custom conv2d" blur --
    deliberately not kornia.filters.gaussian_blur2d, whose sigma-argument
    autograd support is version-dependent."""
    kh, kw = kernel_size
    s = sigma.reshape(-1)  # (n,) -- a 0-dim sigma becomes (1,)
    ax_y = torch.arange(kh, dtype=sigma.dtype, device=sigma.device) - (kh - 1) / 2.0
    ax_x = torch.arange(kw, dtype=sigma.dtype, device=sigma.device) - (kw - 1) / 2.0
    var = (2 * s**2 + 1e-12).view(-1, 1)  # (n, 1)
    gauss_y = torch.exp(-(ax_y**2).view(1, -1) / var)  # (n, kh)
    gauss_x = torch.exp(-(ax_x**2).view(1, -1) / var)  # (n, kw)
    kernel2d = gauss_y[:, :, None] * gauss_x[:, None, :]  # (n, kh, kw)
    kernel2d = kernel2d / kernel2d.sum(dim=(1, 2), keepdim=True)
    return kernel2d.view(-1, 1, kh, kw)


def blur(
    images: Tensor, sigma: Union[float, Tensor], kernel_size: Tuple[int, int] = (67, 67)
) -> Tensor:
    """Gaussian blur; magnitude == sigma directly (unsigned; no lighter/darker
    direction, param_min = 0.0). kernel_size is a fixed hyperparameter, not
    swept -- matches the reference repo. NOTE: this is NOT numerically
    equivalent to augmentations.py's Blur/GaussianBlurPerturbation, which
    instead derive kernel_size from magnitude via cv2's implicit sigma.

    `sigma` may be a scalar (one blur for the whole batch) or a length-B vector
    (a different blur per image)."""
    sigma_t = _as_delta(sigma, images).clamp(min=1e-3)
    b, c = images.shape[0], images.shape[1]
    kernel = _gaussian_kernel2d(kernel_size, sigma_t).to(images.dtype)
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    if kernel.shape[0] == 1:
        # Scalar sigma: one kernel, depthwise over channels.
        kernel = kernel.repeat(c, 1, 1, 1)
        return F.conv2d(images, kernel, padding=padding, groups=c)

    assert kernel.shape[0] == b, (
        f"sigma must be a scalar or a length-{b} vector, got {kernel.shape[0]}"
    )
    # Per-image sigma: fold the batch into the channel dim so every image gets
    # its own kernel, and group by B*C so no kernel sees another image.
    #
    # repeat_interleave, NOT repeat: the folded layout is
    # (b0c0, b0c1, b0c2, b1c0, ...), so image b's kernel must be repeated C
    # times CONSECUTIVELY. `repeat` tiles instead, which would silently apply
    # one image's sigma to a different image's channels -- same shapes, no
    # error, wrong numbers.
    kernel = kernel.repeat_interleave(c, dim=0)  # (b*c, 1, kh, kw)
    out = F.conv2d(
        images.reshape(1, b * c, *images.shape[2:]),
        kernel,
        padding=padding,
        groups=b * c,
    )
    return out.reshape(b, c, *out.shape[2:])


def gaussian_noise(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Additive Gaussian noise scaled by delta (signed; param_min = -eps).
    `delta` may be a scalar or a length-B vector (one scale per image).

    Reparameterized: the gradient flows to the scale, not to the random sample.
    NOTE the sample is drawn fresh on every call, so two calls with the same
    delta differ -- callers that need comparability across calls (the gradient
    probe does) must fix the RNG themselves."""
    delta_t = _as_delta(delta, images)
    d = _broadcast_delta(delta_t, images.shape[0])
    noise = torch.randn(images.shape, dtype=images.dtype, device=images.device)
    return torch.clamp(images + noise * d, 0.0, 1.0)


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
