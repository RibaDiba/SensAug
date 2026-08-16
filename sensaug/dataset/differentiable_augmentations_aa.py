"""Differentiable (autograd-compatible), GPU-batched AutoAugment-family ops.

Companion to sensaug.dataset.differentiable_augmentations, which covers the 8
atomic RGB/HSV/blur/noise ops. This module adds the 9 AutoAugment-family ops the
legacy LEGACY20_OPS vocabulary already exposes as CPU per-image mmseg
transforms -- 5 geometric (rotate, shear x/y, translate x/y) and 4 photometric
(brightness, contrast, sharpness, color) -- as differentiable GPU-batched
functions, so the gradient cross-correlation pipeline can measure
d loss / d magnitude for them the same way it does for the existing 14.

Deliberately a SEPARATE module, not an extension of the sibling one: those ops
are a port of azshue/adversarial_data_augmentation's diffaug.py and stay
faithful to it. These are a different lineage (the AutoAugment op family, via
this repo's own LEGACY20_OPS classes) and are built on kornia's affine and
enhance primitives rather than hand-written math.

Tensor contract (identical to the sibling module, carried over verbatim):
    - dtype: torch.float32
    - shape: (B, 3, H, W)          <- BCHW, batched
    - value range: [0, 1]
    - channel order: RGB (0=R, 1=G, 2=B)

Magnitude contract: every op takes its magnitude as EITHER a scalar (one
magnitude for the whole batch) OR a length-B tensor (one magnitude per image).
The per-image form is what lets one autograd pass measure d loss / d magnitude
separately for each image (sensaug.hooks.grad_hook), and it is the reason this
module never rebuilds a magnitude via `torch.tensor([[...]])`: that silently
DETACHES a tensor argument (verified on torch 2.12 -- it returns
requires_grad=False rather than raising), which would zero the very gradient the
correlation pipeline exists to collect. Magnitudes are assembled with
torch.stack throughout.

Verified against the installed stack (kornia 0.8.2, torch 2.12.1):
    - get_affine_matrix2d returns (B, 3, 3); warp_affine takes M[:, :2, :]
    - its `angle` is in DEGREES, its `sx`/`sy` are shear ANGLES IN RADIANS
      whose resulting matrix element is -tan(s)  (see SHEAR_MAX_FACTOR)
    - adjust_brightness is ADDITIVE (image + factor), NOT torchvision's
      multiplicative form  (see BRIGHTNESS_POS_SCALE)
    - adjust_contrast is a pure multiply, NOT torchvision's mean-blend form
    - factor=1.0 is the identity for contrast, saturation and sharpness, which
      is what makes the `1.0 + delta` centering below correct
"""

from typing import Dict, Optional, Tuple, Union

import torch
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    sharpness,
)
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

# Imported, not duplicated: the two modules must normalize magnitudes onto the
# image's dtype/device identically, or a magnitude that is differentiable in one
# would not be in the other.
#
# The dependency runs ONE WAY -- this module imports the base one, never the
# reverse. That is why the merged registry at the bottom of this file lives here
# rather than there: putting it in the base module would close an import cycle,
# and it would fail only when this module happened to be imported first (which
# its own test file does).
from .differentiable_augmentations import DIFFERENTIABLE_PERTURBATIONS, _as_delta

Tensor = torch.Tensor

# --- magnitude calibration ---------------------------------------------------
#
# Every constant below is MIRRORED from the corresponding
# sensaug.dataset.augmentations LEGACY20_OPS class, so that a magnitude of
# m in [0, 1] means the same visual amount of perturbation here as it does
# there, and the new columns of the correlation matrix R sit on the same footing
# as the existing ones. None of them are chosen freely.
#
# TODO(calibration): mirroring the constant is necessary but NOT sufficient.
# Measured against the legacy classes on a synthetic scene (the render test in
# tests/test_differentiable_augmentations_aa.py, at m=1.0, mean absolute
# difference on the 0-255 scale, alongside how far each op moved the original):
#
#     op            |legacy-ours|   legacy moved   ours moved   verdict
#     translate_x/y      0.00          62-101        62-101      exact
#     sharpness          0.00-0.09      1.0-7.5       1.0-7.6    matches
#     shear_x/y          4.7-5.5       36-48         36-48       matches
#     rotate             8.4-8.5       91            88          matches
#     color_neg         20.8          18.4          22.1         ~20% strong
#     color_pos         23.7          33.2          19.8         ~40% WEAK
#     brightness_neg    21.8          79.4         101.2         ~27% strong
#     contrast_pos      51.0          17.7          49.4         ~2.8x strong
#     brightness_pos    73.7          49.4         123.1         ~2.5x strong
#     contrast_neg      80.5          47.6          79.3         ~1.7x strong
#
# So the 5 geometric ops and sharpness are calibrated -- their residual is
# bilinear-vs-INTER_NEAREST resampling, not scale -- and brightness, contrast
# and color are NOT. Those three differ in FORM, not only in scale, which is why
# mirroring the constant does not fix them:
#
#   1. brightness -- legacy BrightnessTransform uses torchvision's
#      adjust_brightness, which MULTIPLIES (img * factor). kornia's
#      adjust_brightness ADDS (img + factor). At m=1 the legacy op doubles the
#      image; this one drives it to white. Reconciling means either an explicit
#      multiply or a refit scale.
#   2. contrast -- legacy uses torchvision's mean-blend contrast (blend toward
#      the image's own grayscale mean); kornia's adjust_contrast is a pure
#      multiply, which both over-shoots and saturates asymmetrically.
#   3. color -- both scale saturation, but torchvision blends toward grayscale
#      while kornia scales S in HSV, and here the error runs the OTHER way: the
#      positive direction is ~40% too weak, not too strong.
#
# Until those three are refit, their columns of R are on a different footing
# from the rest and should not be compared head-to-head with the existing 14.

# augmentations.Rotate.max_angle -- angle_deg = magnitude * 45. kornia's
# get_affine_matrix2d takes degrees directly, so this transfers unchanged.
ROTATE_MAX_DEG = 45.0

# augmentations.ShearX/ShearY -- shear_factor = magnitude * 0.3, where 0.3 is a
# raw matrix ELEMENT (their get_shear_matrix builds [[1, x, 0], [y, 1, 0]]).
# kornia instead takes a shear ANGLE and puts -tan(angle) in that slot, so the
# wrappers below convert with -atan(factor). That reproduces the legacy matrix
# exactly (checked: -atan(0.3) -> element 0.3000, and the same about-center
# translation term), which a raw `sx=factor` would not -- tan(0.3) = 0.3093 is a
# 3% overshoot, and the sign would be flipped on top of it.
SHEAR_MAX_FACTOR = 0.3

# augmentations.TranslateX/TranslateY -- tx = magnitude * (width / 4.0), i.e. a
# quarter of the image extent, expressed here as the fraction so it can be
# multiplied by whichever extent the op is shifting along.
TRANSLATE_MAX_FRACTION = 0.25

# The photometric legacy classes are deliberately ASYMMETRIC about their
# positive and negative directions -- the negative ones are not sign flips of
# the positive ones, they carry their own smaller scale, so the perturbation
# stays perceptually reasonable in both directions:
#
#   BrightnessTransform      factor = 1.0 + m         -> [1.0, 2.0]
#   NegativeBrightnessTransform      1.0 - 0.6*m      -> [1.0, 0.4]
#   ContrastTransform / NegativeContrastTransform     same 1.0 / -0.6 pair
#   SharpnessTransform / NegativeSharpnessTransform   same 1.0 / -0.6 pair
#   ColorTransform           factor = 1.0 + 2.0*m     -> [1.0, 3.0]
#   NegativeColorTransform          1.0 + 2.0*(-0.3*m) = 1.0 - 0.6*m -> [1.0, 0.4]
#
# so all four share a negative scale of 0.6 while only `color` has a positive
# scale of 2.0. Encoded as (positive, negative) pairs and selected ELEMENTWISE
# on the sign of the magnitude, because with a per-image magnitude the sign can
# differ across the batch -- the same reason hsv_channel's V floor is a
# torch.where and not a python `if`.
BRIGHTNESS_POS_SCALE = 1.0
CONTRAST_POS_SCALE = 1.0
SHARPNESS_POS_SCALE = 1.0
COLOR_POS_SCALE = 2.0
PHOTOMETRIC_NEG_SCALE = 0.6

# Out-of-frame label fill for the geometric ops. 255 is mmseg's ignore index and
# is what the legacy geometric transforms fill their gt_seg_map with
# (`self._shear(results["gt_seg_map"], fill=255)`). Filling with 0 instead --
# kornia's warp_affine default -- would silently relabel every out-of-frame
# pixel as class 0 (road, on Cityscapes) and train against it.
LABEL_IGNORE_INDEX = 255

__all__ = [
    "rotate_op",
    "shear_x_op",
    "shear_y_op",
    "translate_x_op",
    "translate_y_op",
    "brightness_op",
    "contrast_op",
    "sharpness_op",
    "color_op",
    "affine_matrix",
    "geometric_affine_matrix",
    "warp_image_and_label",
    "AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS",
    "DIFF32_OPS",
    "GEOMETRIC_OP_KEYS",
    "PHOTOMETRIC_OP_KEYS",
    "ROTATE_MAX_DEG",
    "SHEAR_MAX_FACTOR",
    "TRANSLATE_MAX_FRACTION",
    "BRIGHTNESS_POS_SCALE",
    "CONTRAST_POS_SCALE",
    "SHARPNESS_POS_SCALE",
    "COLOR_POS_SCALE",
    "PHOTOMETRIC_NEG_SCALE",
    "LABEL_IGNORE_INDEX",
]


def _batch_delta(delta: Union[float, Tensor], images: Tensor) -> Tensor:
    """Normalize a scalar-or-per-image magnitude to a length-B tensor on the
    images' dtype/device, PRESERVING the autograd graph.

    kornia's affine builders want every parameter shaped (B,) -- unlike the
    sibling module's ops, which broadcast a (1, 1, 1, 1) magnitude against the
    image. A scalar is expanded rather than repeated: expand is a view, so the
    gradient of all B uses accumulates back onto the single scalar, which is
    exactly the scalar-magnitude semantics the contract promises.
    """
    d = _as_delta(delta, images).reshape(-1)
    b = images.shape[0]
    if d.shape[0] == 1:
        return d.expand(b)
    assert d.shape[0] == b, (
        f"magnitude must be a scalar or a length-{b} vector (one per image), "
        f"got shape {tuple(d.shape)}"
    )
    return d


def _signed_scale(delta: Tensor, pos_scale: float, neg_scale: float) -> Tensor:
    """Apply a different scale to the positive and negative directions of
    `delta`, elementwise (see the PHOTOMETRIC_*_SCALE block for why the legacy
    classes are asymmetric).

    `neg_scale` is a positive magnitude and multiplies an already-negative
    delta, so `delta = -m` maps to `-neg_scale * m` -- matching e.g.
    NegativeBrightnessTransform's `-0.6 * abs(magnitude)`.
    """
    return torch.where(delta >= 0, delta * pos_scale, delta * neg_scale)


def affine_matrix(
    images: Tensor,
    *,
    angle: Union[float, Tensor] = 0.0,
    tx: Union[float, Tensor] = 0.0,
    ty: Union[float, Tensor] = 0.0,
    sx: Union[float, Tensor] = 0.0,
    sy: Union[float, Tensor] = 0.0,
) -> Tensor:
    """Build the (B, 2, 3) affine matrix for a batch, about each image's center.

    Arguments are in kornia's NATIVE units, not this module's magnitude
    contract: `angle` in degrees, `tx`/`ty` in PIXELS, `sx`/`sy` as shear angles
    in radians. The [0, 1]-magnitude -> native conversion lives in the op
    wrappers, where the calibration constants are. Each argument may be a scalar
    or a length-B tensor and stays differentiable either way.

    Public because the geometric ops cannot be applied to a label map through
    the op functions themselves -- the label needs the SAME matrix, warped with
    nearest-neighbour interpolation and no gradient (see warp_image_and_label).
    """
    b, _, h, w = images.shape
    angle_t = _batch_delta(angle, images)
    tx_t = _batch_delta(tx, images)
    ty_t = _batch_delta(ty, images)
    sx_t = _batch_delta(sx, images)
    sy_t = _batch_delta(sy, images)

    # torch.stack, never torch.tensor([[...]]): the latter detaches a tensor
    # argument (see module docstring) and would make every geometric column of
    # R identically zero.
    translations = torch.stack([tx_t, ty_t], dim=-1)  # (B, 2)
    center = torch.stack(
        [
            torch.full_like(angle_t, w / 2.0),
            torch.full_like(angle_t, h / 2.0),
        ],
        dim=-1,
    )  # (B, 2)
    scale = torch.ones(b, 2, dtype=images.dtype, device=images.device)

    # get_affine_matrix2d returns (B, 3, 3) on kornia 0.8.2 (verified, not
    # assumed -- the shape has moved between versions); warp_affine wants the
    # top two rows.
    matrix = get_affine_matrix2d(translations, center, scale, angle_t, sx=sx_t, sy=sy_t)
    return matrix[:, :2, :]


def _affine_op(
    images: Tensor,
    *,
    angle: Union[float, Tensor] = 0.0,
    tx: Union[float, Tensor] = 0.0,
    ty: Union[float, Tensor] = 0.0,
    sx: Union[float, Tensor] = 0.0,
    sy: Union[float, Tensor] = 0.0,
) -> Tensor:
    """The single primitive behind all five geometric ops: build an affine
    matrix, warp with it. Rotate/shear/translate differ only in which argument
    they populate, so there is exactly one place where the warp can be wrong.

    Bilinear (kornia's default) rather than the legacy transforms' INTER_NEAREST
    -- nearest is piecewise constant, so its derivative w.r.t. the magnitude is
    zero almost everywhere and the whole point of this module would be lost.
    Zero padding matches the legacy `borderValue=0` for images; labels are
    filled with the ignore index instead (warp_image_and_label).
    """
    h, w = images.shape[-2:]
    matrix = affine_matrix(images, angle=angle, tx=tx, ty=ty, sx=sx, sy=sy)
    return warp_affine(images, matrix, dsize=(h, w))


# --- geometric ops -----------------------------------------------------------
#
# Each takes a SIGNED magnitude on this module's [0, 1] contract (so
# delta in [-1, 1]) and converts to kornia's native units with the calibration
# constant mirrored from the legacy class. The registry below bakes the
# direction into the key, matching the sibling module's lighter_/darker_ pattern.


def rotate_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Rotate about the image center. delta=+1 -> +45 degrees
    (augmentations.Rotate at magnitude 1.0)."""
    return _affine_op(images, angle=_batch_delta(delta, images) * ROTATE_MAX_DEG)


def _shear_angle(delta: Union[float, Tensor], images: Tensor) -> Tensor:
    """[0, 1] magnitude -> the kornia shear angle reproducing the legacy shear
    matrix element `delta * SHEAR_MAX_FACTOR`. Negated because kornia writes
    -tan(angle) into that slot; atan because it writes the TANGENT of the angle,
    not the angle."""
    factor = _batch_delta(delta, images) * SHEAR_MAX_FACTOR
    return -torch.atan(factor)


def shear_x_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Horizontal shear about the image center (augmentations.ShearX)."""
    return _affine_op(images, sx=_shear_angle(delta, images))


def shear_y_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Vertical shear about the image center (augmentations.ShearY)."""
    return _affine_op(images, sy=_shear_angle(delta, images))


def translate_x_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Horizontal translation, delta=+1 -> a quarter of the image WIDTH
    (augmentations.TranslateX)."""
    width = images.shape[-1]
    return _affine_op(
        images, tx=_batch_delta(delta, images) * (width * TRANSLATE_MAX_FRACTION)
    )


def translate_y_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Vertical translation, delta=+1 -> a quarter of the image HEIGHT
    (augmentations.TranslateY)."""
    height = images.shape[-2]
    return _affine_op(
        images, ty=_batch_delta(delta, images) * (height * TRANSLATE_MAX_FRACTION)
    )


# --- photometric ops ---------------------------------------------------------
#
# `1.0 + delta` centers "no change" at delta=0, matching the signed-magnitude
# convention the sibling module's RGB/HSV ops already use, and is only valid
# because factor=1.0 is verified to be kornia's identity for contrast,
# saturation and sharpness. brightness is the exception: kornia's is additive,
# so delta is added directly and its identity is delta=0.
#
# Every op clamps to [0, 1] afterwards, matching the sibling module's convention
# (_weighted_rail_perturb and hsv_channel both rail into [0, 1]). The clamp is
# not redundant for saturation/sharpness, which take no clip_output argument.
# Note it does zero the gradient for pixels driven out of range -- intended: a
# saturated pixel genuinely does not respond to more magnitude.


def brightness_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Brightness. NOTE kornia's adjust_brightness is ADDITIVE, unlike the
    legacy class's multiplicative torchvision call -- see TODO(calibration)."""
    d = _signed_scale(
        _batch_delta(delta, images), BRIGHTNESS_POS_SCALE, PHOTOMETRIC_NEG_SCALE
    )
    return adjust_brightness(images, d).clamp(0.0, 1.0)


def contrast_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Contrast. kornia's adjust_contrast is a pure multiply, not the legacy
    class's mean-blend -- see TODO(calibration)."""
    d = _signed_scale(
        _batch_delta(delta, images), CONTRAST_POS_SCALE, PHOTOMETRIC_NEG_SCALE
    )
    return adjust_contrast(images, 1.0 + d).clamp(0.0, 1.0)


def sharpness_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Sharpness (augmentations.SharpnessTransform)."""
    d = _signed_scale(
        _batch_delta(delta, images), SHARPNESS_POS_SCALE, PHOTOMETRIC_NEG_SCALE
    )
    return sharpness(images, 1.0 + d).clamp(0.0, 1.0)


def color_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Saturation (augmentations.ColorTransform -- "color" in AutoAugment's
    naming is saturation). Positive scale is 2.0, not 1.0."""
    d = _signed_scale(
        _batch_delta(delta, images), COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE
    )
    return adjust_saturation(images, 1.0 + d).clamp(0.0, 1.0)


# --- label-safe geometric warp ----------------------------------------------


def warp_image_and_label(
    images: Tensor,
    label: Tensor,
    matrix: Tensor,
    label_fill: float = LABEL_IGNORE_INDEX,
) -> Tuple[Tensor, Tensor]:
    """Apply one affine matrix to an image batch AND its label map.

    Only the 5 geometric ops need this -- photometric ops move no pixels, so
    their labels are unchanged. Three things have to be true at once and none of
    them is the default:

    - the IMAGE warp is bilinear and keeps the graph, so d loss / d magnitude
      still flows;
    - the LABEL warp is nearest and takes `matrix.detach()`, so no gradient ever
      flows through class ids. Gradient through a label is meaningless, and
      interpolating ids would invent classes that do not exist (a bilinear blend
      of id 3 and id 8 is id 5.5);
    - out-of-frame label pixels are filled with the ignore index, not 0. kornia
      pads with zeros by default, which on Cityscapes is the `road` class -- the
      loss would then be computed against a quarter-image of fabricated road.

    `label` may be (B, H, W) or (B, 1, H, W); the returned label matches the
    input's rank and is integral (long).
    """
    h, w = images.shape[-2:]
    images_out = warp_affine(images, matrix, dsize=(h, w), mode="bilinear")

    squeezed = label.dim() == 3
    label_in = label.unsqueeze(1) if squeezed else label
    label_out = warp_affine(
        label_in.to(images.dtype),
        matrix.detach(),
        dsize=label_in.shape[-2:],
        mode="nearest",
        padding_mode="fill",
        fill_value=torch.tensor(
            [float(label_fill)], dtype=images.dtype, device=images.device
        ),
    )
    label_out = label_out.round().long()
    return images_out, label_out.squeeze(1) if squeezed else label_out


def geometric_affine_matrix(
    name: str, images: Tensor, magnitude: Union[float, Tensor]
) -> Optional[Tensor]:
    """The (B, 2, 3) matrix a geometric registry key would warp with, or None if
    `name` is a photometric op (which needs no label warp).

    Exists so a caller holding only a registry KEY -- which is all
    sensaug.hooks.grad_hook has -- can still warp the label alongside the image
    without re-deriving the calibration constants. Keeping this next to the
    wrappers is the point: a constant changed in one and not the other would
    warp the label by a different amount than the image, and nothing downstream
    would notice.
    """
    if name not in GEOMETRIC_OP_KEYS:
        if name in PHOTOMETRIC_OP_KEYS:
            return None
        raise KeyError(
            f"Unknown op {name!r}; expected one of "
            f"{sorted(AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS)}"
        )
    sign = -1.0 if name.endswith("_neg") else 1.0
    delta = _batch_delta(magnitude, images) * sign
    base = name[: -len("_pos")] if name.endswith("_pos") else name[: -len("_neg")]

    if base == "rotate":
        return affine_matrix(images, angle=delta * ROTATE_MAX_DEG)
    if base == "shear_x":
        return affine_matrix(images, sx=_shear_angle(delta, images))
    if base == "shear_y":
        return affine_matrix(images, sy=_shear_angle(delta, images))
    if base == "translate_x":
        return affine_matrix(
            images, tx=delta * (images.shape[-1] * TRANSLATE_MAX_FRACTION)
        )
    return affine_matrix(images, ty=delta * (images.shape[-2] * TRANSLATE_MAX_FRACTION))


# --- registry ----------------------------------------------------------------
#
# Signed pairs, direction baked into the key and magnitude left unsigned --
# mirroring both the sibling module's lighter_R/darker_R pattern and the
# LEGACY20_OPS Rotate/NegativeRotate pattern. Positive and negative are
# separate augmentations rather than one signed axis because the paper treats
# them as such, and because the legacy photometric classes give them different
# scales, so they are genuinely not one another's mirror image.
#
# NOTE the key names are a PROPOSAL. Anything that consumes corr_matrix_log.json
# keys by name will see 18 new ones; confirm the naming with Laura before this
# is merged into the live registry.


def _op(fn, sign: float):
    def _wrapped(images: Tensor, magnitude: Union[float, Tensor]) -> Tensor:
        return fn(images, sign * _as_delta(magnitude, images))

    return _wrapped


AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS: Dict[str, object] = {
    "rotate_pos": _op(rotate_op, +1.0),
    "rotate_neg": _op(rotate_op, -1.0),
    "shear_x_pos": _op(shear_x_op, +1.0),
    "shear_x_neg": _op(shear_x_op, -1.0),
    "shear_y_pos": _op(shear_y_op, +1.0),
    "shear_y_neg": _op(shear_y_op, -1.0),
    "translate_x_pos": _op(translate_x_op, +1.0),
    "translate_x_neg": _op(translate_x_op, -1.0),
    "translate_y_pos": _op(translate_y_op, +1.0),
    "translate_y_neg": _op(translate_y_op, -1.0),
    "brightness_pos": _op(brightness_op, +1.0),
    "brightness_neg": _op(brightness_op, -1.0),
    "contrast_pos": _op(contrast_op, +1.0),
    "contrast_neg": _op(contrast_op, -1.0),
    "sharpness_pos": _op(sharpness_op, +1.0),
    "sharpness_neg": _op(sharpness_op, -1.0),
    "color_pos": _op(color_op, +1.0),
    "color_neg": _op(color_op, -1.0),
}

GEOMETRIC_OP_KEYS = frozenset(
    f"{base}_{sign}"
    for base in ("rotate", "shear_x", "shear_y", "translate_x", "translate_y")
    for sign in ("pos", "neg")
)
PHOTOMETRIC_OP_KEYS = frozenset(
    f"{base}_{sign}"
    for base in ("brightness", "contrast", "sharpness", "color")
    for sign in ("pos", "neg")
)

assert GEOMETRIC_OP_KEYS | PHOTOMETRIC_OP_KEYS == set(
    AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS
), "geometric/photometric key split has drifted from the registry"


# --- the merged vocabulary the grad_corr pipeline sweeps ----------------------
#
# 32 keys: the base module's 14 (Shu et al.'s RGB/HSV/blur/noise ops) plus the 18
# above. This is what `--aug-type=grad_corr` measures -- CollectGradientHook's
# sweep, the axes of R in PerturbationSensitivityAnalysisHookWithGradients, and
# the DIFF_PERTURBATIONS pipeline wrappers in sensaug.dataset.augmentations all
# read this name, so adding a key here adds a row and a column to R.
#
# DIFFERENTIABLE_PERTURBATIONS itself is left at 14 on purpose: it means "the
# ops ported from the reference repo", and the tests that pin it to exactly that
# vocabulary are still testing something true.
#
# WARNING -- the 10 geometric keys move pixels, and neither consumer moves the
# LABEL to match:
#   * sensaug.dataset.augmentations._DiffAugTransform.transform rewrites
#     results["img"] only, never results["gt_seg_map"];
#   * CollectGradientHook._grad_for_op computes model.loss(perturbed,
#     data_samples) against the unwarped data_samples.
# So for rotate/shear/translate the measured d loss / d magnitude is dominated
# by image-label misalignment, not by the model's sensitivity to the
# perturbation. The numbers will look plausible and large. warp_image_and_label
# above is the piece that fixes this; wiring it into those two call sites is
# deliberately not done here.
DIFF32_OPS: Dict[str, object] = {
    **DIFFERENTIABLE_PERTURBATIONS,
    **AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS,
}

assert len(DIFF32_OPS) == len(DIFFERENTIABLE_PERTURBATIONS) + len(
    AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS
), "a new op name collides with one of the base module's -- the merge would shadow it"
