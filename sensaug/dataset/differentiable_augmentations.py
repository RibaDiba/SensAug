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
import warnings
from typing import Tuple, Union

import numpy as np
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

# Default Gaussian blur support, in taps. Sized to this module's [0, 1]
# magnitude contract, NOT picked freely: for `blur` the magnitude IS sigma, and
# both magnitude sources clamp to [0, 1] (adaptive_sensitivity_analysis_new's
# min_level/max_level = 0.0/1.0, and corr_magnitudes' clip), so 13 taps span
# +-6 sigma at the largest sigma reachable through DIFFERENTIABLE_PERTURBATIONS.
#
# This replaces a previous default of (67, 67). Over the whole reachable range
# the two are numerically indistinguishable -- measured against the 67-tap
# kernel, the difference is exactly zero for sigma <= 0.5 and 2.4e-7 (float32
# round-off) at sigma = 1.0 -- but 67 taps cost 26x more per pixel, which is
# what made the CPU per-image SA round-eval path (sensaug.loops.sensaug_loop.RobustValLoop
# -> the Diff* transforms in sensaug.dataset.augmentations) take hours per
# round. Together with the separable convolution in `blur`, this is ~240x
# faster per full-resolution image with no change to the numbers.
#
# Callers needing a larger sigma must pass a larger kernel_size explicitly;
# `blur` warns when sigma outruns the kernel it was given.
BLUR_KERNEL_SIZE = (13, 13)

__all__ = [
    "color_channel",
    "hsv_channel",
    "blur",
    "gaussian_noise",
    "DiffAugment",
    "BLUR_KERNEL_SIZE",
    "DIFFERENTIABLE_PERTURBATIONS",
    "img_to_rgb01",
    "rgb01_to_img",
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


def _gaussian_kernel1d(size: int, sigma: Tensor) -> Tensor:
    """Hand-built (guaranteed differentiable w.r.t. `sigma`) 1D Gaussian kernel
    STACK, shape (n, size) with n = 1 for a scalar sigma and n = B for a
    per-image sigma. Mirrors the reference repo's own "custom conv2d" blur --
    deliberately not kornia.filters.gaussian_blur2d, whose sigma-argument
    autograd support is version-dependent.

    ONE DIMENSION, not two: an isotropic 2D Gaussian is separable, and this
    function's previous 2D form built the kernel as exactly that -- the outer
    product `gauss_y[:, :, None] * gauss_x[:, None, :]`, normalized by its 2D
    sum. Since sum(outer(a, b)) == sum(a) * sum(b), normalizing each 1D factor
    by its own sum reproduces the identical 2D kernel, so `blur` can convolve
    with the two factors in sequence instead of materializing their product.
    That is an algebraic identity, not an approximation: the measured
    difference against the dense 2D convolution is float32 round-off (~1e-6 on
    values in [0, 1]), and it turns a kh*kw-tap convolution into kh+kw taps --
    a 33x reduction in multiply-adds at the default kernel size, which is what
    makes the CPU per-image SA path affordable (see `blur`).
    """
    s = sigma.reshape(-1)  # (n,) -- a 0-dim sigma becomes (1,)
    ax = torch.arange(size, dtype=sigma.dtype, device=sigma.device) - (size - 1) / 2.0
    var = (2 * s**2 + 1e-12).view(-1, 1)  # (n, 1)
    gauss = torch.exp(-(ax**2).view(1, -1) / var)  # (n, size)
    return gauss / gauss.sum(dim=1, keepdim=True)


def _warn_if_undersupported(sigma: Tensor, kernel_size: Tuple[int, int]) -> None:
    """Warn when `kernel_size` is too small to resolve `sigma`.

    BLUR_KERNEL_SIZE is sized for this module's [0, 1] magnitude contract, so a
    caller reaching outside that contract would otherwise silently get a
    truncated -- effectively weaker -- blur than the sigma it asked for. Warn
    rather than raise: passing a deliberately under-supported kernel is a valid
    thing to do in tests (it makes per-image kernel mix-ups easier to detect),
    and no production path can trip this, since both magnitude sources clamp to
    [0, 1] (sensitivity_analysis.adaptive_sensitivity_analysis_new's
    min_level/max_level, and corr_magnitudes' clip).

    Reads sigma under no_grad: this is a diagnostic on the magnitude's value and
    must not become part of what autograd differentiates.
    """
    with torch.no_grad():
        sigma_max = float(sigma.max())
    # A Gaussian is numerically dead beyond ~3 sigma, so the kernel's half-width
    # (k - 1) / 2 taps must cover 3 * sigma.
    supported = (min(kernel_size) - 1) / 6.0
    if sigma_max > supported:
        warnings.warn(
            f"blur: sigma up to {sigma_max:.3g} exceeds what kernel_size="
            f"{kernel_size} resolves (max ~{supported:.3g} at 3 sigma); the "
            f"Gaussian is truncated, so the blur is weaker than requested. Pass "
            f"a larger kernel_size (>= {2 * math.ceil(3 * sigma_max) + 1}).",
            RuntimeWarning,
            stacklevel=3,
        )


def blur(
    images: Tensor,
    sigma: Union[float, Tensor],
    kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE,
) -> Tensor:
    """Gaussian blur; magnitude == sigma directly (unsigned; no lighter/darker
    direction, param_min = 0.0). kernel_size is a fixed hyperparameter, not
    swept -- matches the reference repo. NOTE: this is NOT numerically
    equivalent to augmentations.py's Blur/GaussianBlurPerturbation, which
    instead derive kernel_size from magnitude via cv2's implicit sigma.

    `sigma` may be a scalar (one blur for the whole batch) or a length-B vector
    (a different blur per image).

    Convolves with the two 1D factors of the separable Gaussian in sequence
    rather than with their dense 2D product -- same kernel, kh+kw taps instead
    of kh*kw (see _gaussian_kernel1d). kernel_size is deliberately NOT derived
    from sigma: a sigma-dependent kernel size would, for a per-image sigma, make
    the whole batch share a size chosen by its largest sigma, and image i's
    output would then depend on the other images' magnitudes -- exactly the
    cross-image coupling CollectGradientHook._sweep is built to exclude.
    """
    sigma_t = _as_delta(sigma, images).clamp(min=1e-3)
    _warn_if_undersupported(sigma_t, kernel_size)
    b, c = images.shape[0], images.shape[1]
    kh, kw = kernel_size
    kernel_y = _gaussian_kernel1d(kh, sigma_t).to(images.dtype)
    kernel_x = _gaussian_kernel1d(kw, sigma_t).to(images.dtype)

    if kernel_y.shape[0] == 1:
        # Scalar sigma: one kernel, depthwise over channels.
        kernel_y = kernel_y.view(1, 1, kh, 1).repeat(c, 1, 1, 1)
        kernel_x = kernel_x.view(1, 1, 1, kw).repeat(c, 1, 1, 1)
        out = F.conv2d(images, kernel_y, padding=(kh // 2, 0), groups=c)
        return F.conv2d(out, kernel_x, padding=(0, kw // 2), groups=c)

    assert kernel_y.shape[0] == b, (
        f"sigma must be a scalar or a length-{b} vector, got {kernel_y.shape[0]}"
    )
    # Per-image sigma: fold the batch into the channel dim so every image gets
    # its own kernel, and group by B*C so no kernel sees another image.
    #
    # repeat_interleave, NOT repeat: the folded layout is
    # (b0c0, b0c1, b0c2, b1c0, ...), so image b's kernel must be repeated C
    # times CONSECUTIVELY. `repeat` tiles instead, which would silently apply
    # one image's sigma to a different image's channels -- same shapes, no
    # error, wrong numbers.
    kernel_y = kernel_y.view(b, 1, kh, 1).repeat_interleave(c, dim=0)  # (b*c, 1, kh, 1)
    kernel_x = kernel_x.view(b, 1, 1, kw).repeat_interleave(c, dim=0)  # (b*c, 1, 1, kw)
    folded = images.reshape(1, b * c, *images.shape[2:])
    out = F.conv2d(folded, kernel_y, padding=(kh // 2, 0), groups=b * c)
    out = F.conv2d(out, kernel_x, padding=(0, kw // 2), groups=b * c)
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

    def __init__(self, eps: float = 1.0, kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE):
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


# --- mmseg image <-> tensor conversion ---------------------------------------
#
# Used by the pipeline wrappers in sensaug.dataset.augmentations, which is where
# the registered transform classes and the DIFF_PERTURBATIONS registry live --
# NOT here. This module stays free of any mmcv/mmseg dependency so the ops and
# their gradients remain testable without the OpenMMLab stack installed. Only
# the conversion math lives here, so that it is testable on the same terms.


def img_to_rgb01(img: np.ndarray) -> Tensor:
    """mmseg's HWC uint8 BGR image -> this module's (1, 3, H, W) float32 RGB [0, 1].

    Mirrors what the gradient probe reconstructs in grad_hook._probe_batch (which
    de-normalizes the data preprocessor's output back to RGB [0, 1]), so the SA
    pipeline and the probe hand the ops the same tensor for the same image.
    """
    rgb = np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return tensor / 255.0


def rgb01_to_img(tensor: Tensor) -> np.ndarray:
    """Inverse of img_to_rgb01.

    Rounds rather than truncates: `.to(torch.uint8)` truncates, which would bias
    every channel down by up to one level and make magnitude=0 a visible darkening
    rather than the exact no-op it must be.
    """
    rgb = (tensor.squeeze(0).permute(1, 2, 0) * 255.0).round().clamp(0.0, 255.0)
    return np.ascontiguousarray(rgb.to(torch.uint8).numpy()[..., ::-1])  # -> BGR
