"""Tests for the AutoAugment-family differentiable ops.

Deliberately runs BEFORE any of this is wired into sensaug.hooks.grad_hook: a
broken autograd graph here is invisible downstream (the correlation matrix just
grows a column of zeros, or worse, of plausible-looking noise) and only shows up
after a job has already burned Della time. The gradient tests below are the
cheap check that catches it first.

Like tests/test_differentiable_augmentations.py, this suite avoids importing the
mmcv/mmseg/cv2 stack. The one exception is the magnitude comparison at the
bottom, which needs the legacy transforms by definition -- it is skipped when
that stack is absent, and it ASSERTS NOTHING: it writes a side-by-side image for
a human to look at.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.dataset.differentiable_augmentations_aa import (  # noqa: E402
    AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS,
    BRIGHTNESS_POS_SCALE,
    COLOR_POS_SCALE,
    CONTRAST_POS_SCALE,
    GEOMETRIC_OP_KEYS,
    LABEL_IGNORE_INDEX,
    PHOTOMETRIC_NEG_SCALE,
    PHOTOMETRIC_OP_KEYS,
    ROTATE_MAX_DEG,
    SHARPNESS_POS_SCALE,
    SHEAR_MAX_FACTOR,
    TRANSLATE_MAX_FRACTION,
    _signed_scale,
    affine_matrix,
    brightness_op,
    color_op,
    contrast_op,
    geometric_affine_matrix,
    rotate_op,
    sharpness_op,
    shear_x_op,
    shear_y_op,
    translate_x_op,
    translate_y_op,
    warp_image_and_label,
)

EXPECTED_KEYS = {
    "rotate_pos", "rotate_neg",
    "shear_x_pos", "shear_x_neg",
    "shear_y_pos", "shear_y_neg",
    "translate_x_pos", "translate_x_neg",
    "translate_y_pos", "translate_y_neg",
    "brightness_pos", "brightness_neg",
    "contrast_pos", "contrast_neg",
    "sharpness_pos", "sharpness_neg",
    "color_pos", "color_neg",
}

ALL_OPS = [
    rotate_op, shear_x_op, shear_y_op, translate_x_op, translate_y_op,
    brightness_op, contrast_op, sharpness_op, color_op,
]


@pytest.fixture
def batch():
    """Textured, not flat: several of these ops (sharpness above all) are
    identity on a constant image, so a flat fixture would pass every gradient
    test while measuring nothing."""
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def big_batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 32, 32)


# --- Phase 5: vocabulary -----------------------------------------------------


def test_registry_exposes_the_18_signed_keys():
    """Mirrors test_registry_covers_legacy_perturbation_vocabulary in the
    sibling suite: the 9 AutoAugment-family ops as signed pairs."""
    assert set(AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS) == EXPECTED_KEYS
    assert len(AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS) == 18


def test_registry_keys_are_disjoint_from_the_sibling_vocabulary():
    """These 18 are meant to EXTEND the existing 14, so a name collision would
    silently shadow an op in whichever dict is merged second."""
    from sensaug.dataset.differentiable_augmentations import (
        DIFFERENTIABLE_PERTURBATIONS,
    )

    assert not (
        set(AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS) & set(DIFFERENTIABLE_PERTURBATIONS)
    )


def test_geometric_photometric_split_partitions_the_registry():
    assert GEOMETRIC_OP_KEYS.isdisjoint(PHOTOMETRIC_OP_KEYS)
    assert GEOMETRIC_OP_KEYS | PHOTOMETRIC_OP_KEYS == EXPECTED_KEYS
    assert len(GEOMETRIC_OP_KEYS) == 10 and len(PHOTOMETRIC_OP_KEYS) == 8


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_registry_op_shape_and_finite(batch, name):
    out = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](batch, 0.3)
    assert out.shape == batch.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_registry_ops_stay_in_the_unit_range(batch, name):
    """The module contract says [0, 1] in and [0, 1] out; the next op in a
    composition, and the model's own normalization, both assume it."""
    out = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](batch, 0.8)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_zero_magnitude_is_the_identity(batch, name):
    """Magnitude 0 must be an exact no-op, not merely a small one -- SA sweeps
    start at 0, and a biased zero point would offset every level above it.
    Geometric ops go through a resampling grid, hence the tolerance."""
    out = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](batch, 0.0)
    assert torch.allclose(out, batch, atol=1e-5)


@pytest.mark.parametrize("base", sorted({k.rsplit("_", 1)[0] for k in EXPECTED_KEYS}))
def test_pos_and_neg_move_in_opposite_directions(batch, base):
    """The signed pair is the whole reason there are 18 keys and not 9: if
    _pos and _neg produced the same image, half the registry would be
    duplicate columns in R."""
    pos = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[f"{base}_pos"](batch, 0.5)
    neg = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[f"{base}_neg"](batch, 0.5)
    assert not torch.allclose(pos, neg, atol=1e-4)


# --- Phase 7: gradient smoke tests -------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_gradient_flows_to_the_magnitude(batch, name):
    """THE test this file exists for. A stray .detach(), a magnitude rebuilt
    with torch.tensor([[...]]), or a nearest-neighbour warp would each leave the
    forward pass looking perfect and the gradient dead -- and d loss/d magnitude
    is the only thing the correlation pipeline reads."""
    magnitude = torch.tensor(0.4, requires_grad=True)
    out = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](batch, magnitude)

    grad = torch.autograd.grad(out.sum(), magnitude, allow_unused=True)[0]
    assert grad is not None, f"{name}: magnitude is not part of the graph"
    assert torch.isfinite(grad).all()
    assert float(grad.abs()) > 0.0, f"{name}: gradient is exactly zero"


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_per_image_magnitude_gives_a_per_image_gradient(big_batch, name):
    """The per-image form is what lets one probe pass measure every image
    separately (grad_hook). A magnitude that got collapsed to a scalar
    somewhere would still backward fine -- it would just return one number, and
    the per-image structure R is built on would be gone."""
    magnitudes = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    out = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](big_batch, magnitudes)

    grad = torch.autograd.grad(out.sum(), magnitudes)[0]
    assert grad.shape == (4,)
    assert torch.isfinite(grad).all()
    assert (grad.abs() > 0).all(), f"{name}: some images got no gradient"


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_each_image_responds_only_to_its_own_magnitude(big_batch, name):
    """Cross-image coupling is the failure mode CollectGradientHook._sweep is
    built to exclude: perturb image 0's magnitude only, and images 1..3 must be
    bit-identical to a run where it was never touched."""
    base = torch.tensor([0.3, 0.3, 0.3, 0.3])
    bumped = base.clone()
    bumped[0] = 0.7

    out_base = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](big_batch, base)
    out_bumped = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](big_batch, bumped)

    assert torch.allclose(out_base[1:], out_bumped[1:], atol=1e-6)
    assert not torch.allclose(out_base[0], out_bumped[0], atol=1e-4)


@pytest.mark.parametrize("op", ALL_OPS, ids=lambda f: f.__name__)
def test_scalar_and_broadcast_magnitudes_agree(batch, op):
    """A scalar magnitude is expanded, not repeated; the two forms must produce
    identical pixels or the scalar path and the probe path would disagree."""
    scalar = op(batch, 0.35)
    per_image = op(batch, torch.tensor([0.35, 0.35]))
    assert torch.allclose(scalar, per_image, atol=1e-6)


def test_scalar_magnitude_accumulates_gradient_over_the_batch(batch):
    """Expanding rather than repeating means one scalar receives the sum of the
    batch's gradients -- pin it, since `expand` vs `repeat` here is invisible in
    the forward pass."""
    scalar = torch.tensor(0.4, requires_grad=True)
    per_image = torch.tensor([0.4, 0.4], requires_grad=True)

    g_scalar = torch.autograd.grad(rotate_op(batch, scalar).sum(), scalar)[0]
    g_each = torch.autograd.grad(rotate_op(batch, per_image).sum(), per_image)[0]

    assert torch.allclose(g_scalar, g_each.sum(), rtol=1e-4)


def test_rejects_a_magnitude_that_is_neither_scalar_nor_length_b(batch):
    with pytest.raises(AssertionError, match="length-2 vector"):
        rotate_op(batch, torch.tensor([0.1, 0.2, 0.3]))


# --- calibration constants: pinned to the legacy classes ---------------------
#
# These do not check the perturbation looks right -- that is the eyeball test at
# the bottom. They check that the CONSTANT has not drifted from the legacy value
# it was mirrored from, so a future edit cannot silently put the new columns of
# R on a different scale from the old ones.


def test_calibration_constants_match_the_legacy_classes():
    assert ROTATE_MAX_DEG == 45.0  # augmentations.Rotate.max_angle
    assert SHEAR_MAX_FACTOR == 0.3  # augmentations.ShearX: magnitude * 0.3
    assert TRANSLATE_MAX_FRACTION == 0.25  # augmentations.TranslateX: dim / 4.0


def test_affine_matrix_with_no_arguments_is_the_identity(batch):
    """The primitive every geometric op routes through: with nothing set it must
    be exactly the identity, or every op inherits a constant offset."""
    matrix = affine_matrix(batch)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).expand(
        batch.shape[0], 2, 3
    )
    assert matrix.shape == (batch.shape[0], 2, 3)
    assert torch.allclose(matrix, identity, atol=1e-6)


def test_shear_matrix_reproduces_the_legacy_matrix_element(batch):
    """The legacy shear matrix puts the raw factor in M[0, 1]; kornia's shear
    argument is an ANGLE and lands -tan(angle) there. This pins the -atan()
    conversion -- passing the factor directly would overshoot by tan(x)/x AND
    flip the sign, and both errors are invisible without a matrix comparison."""
    magnitude = 0.6
    matrix = geometric_affine_matrix("shear_x_pos", batch, magnitude)
    assert matrix is not None
    assert matrix[0, 0, 1].item() == pytest.approx(magnitude * SHEAR_MAX_FACTOR, abs=1e-6)

    matrix_y = geometric_affine_matrix("shear_y_pos", batch, magnitude)
    assert matrix_y is not None
    assert matrix_y[0, 1, 0].item() == pytest.approx(
        magnitude * SHEAR_MAX_FACTOR, abs=1e-6
    )


def test_translate_matrix_is_a_quarter_extent_at_full_magnitude(batch):
    """augmentations.TranslateX shifts by magnitude * (width / 4.0) pixels."""
    matrix = geometric_affine_matrix("translate_x_pos", batch, 1.0)
    assert matrix is not None
    assert matrix[0, 0, 2].item() == pytest.approx(batch.shape[-1] / 4.0, abs=1e-4)


def test_rotate_matrix_is_45_degrees_at_full_magnitude(batch):
    import math

    matrix = geometric_affine_matrix("rotate_pos", batch, 1.0)
    assert matrix is not None
    assert matrix[0, 0, 0].item() == pytest.approx(math.cos(math.radians(45.0)), abs=1e-5)


def test_photometric_negative_direction_uses_the_smaller_legacy_scale():
    """The legacy negative classes are NOT sign flips -- they carry their own
    0.6 scale (NegativeBrightnessTransform's `-0.6 * abs(magnitude)`), while
    only `color` has a positive scale other than 1.0 (ColorTransform's
    `1.0 + magnitude * 2.0`).

    Checked on the scale itself rather than on the resulting pixels: the ops
    clamp to [0, 1], and clamping can make the smaller-scaled direction move
    pixels FARTHER -- see the contrast test below. A pixel-level assertion here
    would be measuring saturation, not calibration."""
    d = torch.tensor([0.5, -0.5])

    for pos_scale in (BRIGHTNESS_POS_SCALE, CONTRAST_POS_SCALE, SHARPNESS_POS_SCALE):
        assert pos_scale == 1.0
        assert torch.allclose(
            _signed_scale(d, pos_scale, PHOTOMETRIC_NEG_SCALE),
            torch.tensor([0.5, -0.3]),
        )

    # ColorTransform doubles: 1.0 + 2.0*m positive, but NegativeColorTransform's
    # -0.3 scale times that same 2.0 lands on the shared -0.6 negative slope.
    assert COLOR_POS_SCALE == 2.0
    assert torch.allclose(
        _signed_scale(d, COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE),
        torch.tensor([1.0, -0.3]),
    )


def test_signed_scale_selects_per_image_not_per_batch(big_batch):
    """With a per-image magnitude the sign varies WITHIN the batch, so the
    positive/negative scale has to be chosen elementwise -- a python `if` on the
    sign would silently apply one image's direction to the whole batch."""
    d = torch.tensor([0.5, -0.5, 0.5, -0.5])
    scaled = _signed_scale(d, COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    assert torch.allclose(scaled, torch.tensor([1.0, -0.3, 1.0, -0.3]))


def _mean_pixel_shift(op, batch, delta):
    return float((op(batch, delta) - batch).abs().mean())


@pytest.mark.parametrize("op", [contrast_op, color_op], ids=["contrast", "color"])
def test_multiplicative_ops_invert_under_the_unit_clamp(batch, op):
    """A larger scale does NOT mean a larger pixel shift for the two ops that
    scale multiplicatively away from a fixed point (kornia's adjust_contrast is
    a pure multiply; adjust_saturation scales S in HSV). At magnitude 0.9 the
    positive direction pushes roughly half the pixels past 1.0, where the [0, 1]
    clamp truncates the shift -- so the 0.6-scaled negative direction, which
    contracts toward the fixed point and cannot leave the range, moves pixels
    FARTHER.

    Pinned as known behaviour, not as desirable: it is exactly why the
    calibration constants cannot be validated by comparing pixel displacements,
    and it is one of the form mismatches TODO(calibration) is about (the legacy
    ContrastTransform uses torchvision's mean-blend, which has no such
    blow-up)."""
    assert _mean_pixel_shift(op, batch, -0.9) > _mean_pixel_shift(op, batch, 0.9)


@pytest.mark.parametrize(
    "op", [brightness_op, sharpness_op], ids=["brightness", "sharpness"]
)
def test_additive_and_blend_ops_do_not_invert(batch, op):
    """The counterpart: brightness shifts by a constant and sharpness blends
    toward a filtered copy, so neither amplifies out of range the way a multiply
    does, and the smaller negative scale does show up as a smaller shift."""
    assert _mean_pixel_shift(op, batch, -0.9) < _mean_pixel_shift(op, batch, 0.9)


def test_geometric_affine_matrix_returns_none_for_photometric_ops(batch):
    """Photometric ops move no pixels, so a caller asking for their label warp
    should get an explicit None rather than an identity matrix it would then
    pointlessly resample the label with."""
    for name in sorted(PHOTOMETRIC_OP_KEYS):
        assert geometric_affine_matrix(name, batch, 0.5) is None


def test_geometric_affine_matrix_rejects_unknown_names(batch):
    with pytest.raises(KeyError):
        geometric_affine_matrix("not_an_op", batch, 0.5)


@pytest.mark.parametrize("name", sorted(GEOMETRIC_OP_KEYS))
def test_geometric_affine_matrix_agrees_with_the_op_it_names(batch, name):
    """The matrix handed to the label must be the SAME one the image was warped
    with. If the two ever drift apart, image and label are warped by different
    amounts and nothing downstream can detect it."""
    from kornia.geometry.transform import warp_affine

    magnitude = 0.5
    matrix = geometric_affine_matrix(name, batch, magnitude)
    assert matrix is not None

    via_matrix = warp_affine(batch, matrix, dsize=batch.shape[-2:])
    via_op = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[name](batch, magnitude)
    assert torch.allclose(via_matrix, via_op, atol=1e-6)


# --- Phase 6: label-safe warp ------------------------------------------------


def test_warp_image_and_label_keeps_image_gradient_and_drops_label_gradient(batch):
    magnitude = torch.tensor(0.5, requires_grad=True)
    matrix = geometric_affine_matrix("rotate_pos", batch, magnitude)
    label = torch.randint(0, 19, (batch.shape[0], batch.shape[-2], batch.shape[-1]))

    img_out, label_out = warp_image_and_label(batch, label, matrix)

    assert img_out.requires_grad
    assert not label_out.is_floating_point()
    grad = torch.autograd.grad(img_out.sum(), magnitude)[0]
    assert float(grad.abs()) > 0.0


def test_warp_image_and_label_invents_no_class_ids(batch):
    """Nearest, never bilinear: a bilinear blend of class 3 and class 8 is class
    5.5, which rounds to a class that was never in the image."""
    label = torch.randint(0, 19, (batch.shape[0], batch.shape[-2], batch.shape[-1]))
    matrix = geometric_affine_matrix("rotate_pos", batch, 0.4)

    _img_out, label_out = warp_image_and_label(batch, label, matrix)
    produced = set(label_out.unique().tolist()) - {LABEL_IGNORE_INDEX}
    assert produced <= set(label.unique().tolist())


def test_warp_image_and_label_fills_out_of_frame_with_the_ignore_index(batch):
    """Filling with kornia's default 0 would relabel every out-of-frame pixel as
    Cityscapes `road` and compute a real loss against it."""
    label = torch.full((batch.shape[0], batch.shape[-2], batch.shape[-1]), 7)
    matrix = geometric_affine_matrix("translate_x_pos", batch, 1.0)

    _img_out, label_out = warp_image_and_label(batch, label, matrix)
    assert LABEL_IGNORE_INDEX in set(label_out.unique().tolist())
    assert 0 not in set(label_out.unique().tolist())


def test_warp_image_and_label_preserves_label_rank(batch):
    """(B, H, W) in -> (B, H, W) out; (B, 1, H, W) in -> (B, 1, H, W) out.
    mmseg hands these around in both shapes."""
    b, _, h, w = batch.shape
    matrix = geometric_affine_matrix("shear_y_neg", batch, 0.3)

    _i, out3 = warp_image_and_label(batch, torch.randint(0, 19, (b, h, w)), matrix)
    _i, out4 = warp_image_and_label(batch, torch.randint(0, 19, (b, 1, h, w)), matrix)
    assert out3.shape == (b, h, w)
    assert out4.shape == (b, 1, h, w)


# --- magnitude sanity check: RENDERS, DOES NOT ASSERT ------------------------
#
# Phase 4 is not finished and this test does not pretend otherwise. Comparing
# the new ops to the legacy NEW_PERTURBATIONS classes programmatically would
# just encode whatever mismatch exists today as the expected answer -- and three
# of the photometric ops are known to differ in FORM, not only in scale (kornia's
# additive brightness vs torchvision's multiplicative, kornia's pure-multiply
# contrast vs torchvision's mean-blend). So: render both at matched magnitudes,
# write the strip to disk, and let a human decide where they diverge.
#
# Opt-in via an env var rather than a custom pytest marker, so this file needs
# no change to pyproject.toml's `markers` list. Run it explicitly:
#     AA_COMPARE=1 pytest tests/test_differentiable_augmentations_aa.py -k render -s
# and point AA_COMPARE_DIR somewhere durable to keep the strips.

LEGACY_COUNTERPART = {
    "rotate_pos": "Rotate",
    "rotate_neg": "NegativeRotate",
    "shear_x_pos": "ShearX",
    "shear_x_neg": "NegativeShearX",
    "shear_y_pos": "ShearY",
    "shear_y_neg": "NegativeShearY",
    "translate_x_pos": "TranslateX",
    "translate_x_neg": "NegativeTranslateX",
    "translate_y_pos": "TranslateY",
    "translate_y_neg": "NegativeTranslateY",
    "brightness_pos": "BrightnessTransform",
    "brightness_neg": "NegativeBrightnessTransform",
    "contrast_pos": "ContrastTransform",
    "contrast_neg": "NegativeContrastTransform",
    "sharpness_pos": "SharpnessTransform",
    "sharpness_neg": "NegativeSharpnessTransform",
    "color_pos": "ColorTransform",
    "color_neg": "NegativeColorTransform",
}


@pytest.mark.requires_mmseg
@pytest.mark.skipif(
    not os.environ.get("AA_COMPARE"),
    reason="visual calibration check; set AA_COMPARE=1 to render the strips",
)
def test_render_magnitude_comparison_against_legacy_classes(tmp_path):
    """Writes one PNG per op: legacy transform on the left, this module's op on
    the right, at magnitudes 0.25 / 0.5 / 1.0. Always passes -- it produces
    something to look at, it does not judge."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    legacy = pytest.importorskip("sensaug.dataset.augmentations")
    from sensaug.dataset.differentiable_augmentations import img_to_rgb01, rgb01_to_img

    out_dir = os.environ.get("AA_COMPARE_DIR", str(tmp_path))
    os.makedirs(out_dir, exist_ok=True)

    # A synthetic scene with a colour ramp, hard edges, flat regions AND a
    # high-frequency patch. The last one is not decoration: sharpness only acts
    # on detail, so without it sharpness moves almost no pixels and its row of
    # the report reads ~0.00 no matter how badly calibrated it is.
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (xx * 255 // w).astype(np.uint8)
    img[..., 1] = (yy * 255 // h).astype(np.uint8)
    img[..., 2] = 128
    img[32:96, 32:96] = 240
    img[56:72, 8:120] = 16
    checker = (((xx // 2) + (yy // 2)) % 2).astype(np.uint8) * 255
    img[8:56, 72:120] = checker[8:56, 72:120, None]

    for key, legacy_name in sorted(LEGACY_COUNTERPART.items()):
        legacy_cls = getattr(legacy, legacy_name, None)
        if legacy_cls is None:
            print(f"[skip] no legacy class {legacy_name} for {key}")
            continue

        rows = []
        for magnitude in (0.25, 0.5, 1.0):
            legacy_out = legacy_cls(magnitude=magnitude)(
                {"img": np.ascontiguousarray(img.copy())}
            )["img"]
            ours = AUTOAUGMENT_DIFFERENTIABLE_PERTURBATIONS[key](
                img_to_rgb01(img), magnitude
            )
            ours_out = rgb01_to_img(ours)
            gap = np.full((h, 4, 3), 255, dtype=np.uint8)
            rows.append(np.hstack([legacy_out, gap, ours_out]))

            # The disagreement alone is not readable: "legacy and ours differ by
            # 0.1" means they agree if both moved the image by 40, and means
            # neither op did anything if both moved it by 0.1. Report how far
            # each moved the ORIGINAL alongside it.
            base = img.astype(float)
            print(
                f"{key:>16} m={magnitude:<5} "
                f"mean|legacy-ours| = "
                f"{np.abs(legacy_out.astype(float) - ours_out.astype(float)).mean():7.2f}"
                f"   (legacy moved {np.abs(legacy_out.astype(float) - base).mean():6.2f},"
                f" ours moved {np.abs(ours_out.astype(float) - base).mean():6.2f})"
                f"  -- for eyeballing, NOT asserted on"
            )

        path = os.path.join(out_dir, f"{key}__vs__{legacy_name}.png")
        cv2.imwrite(path, np.vstack(rows))
        print(f"  wrote {path}")

    print(f"\nComparison strips written to {out_dir} -- left: legacy, right: ours.")
