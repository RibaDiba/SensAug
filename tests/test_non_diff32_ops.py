"""Tests for NON_DIFF32_OPS -- the 32 ops R is computed over, played through the
plain CPU transform classes instead of the differentiable ones.

No --aug-type routes here any more: ours/default/grad_corr all run on "diff32",
the GPU-batched differentiable ops. This set is kept as the CPU reference the
equivalence checks compare against, and it stays selectable for a deliberate CPU
run. What these tests pin is that it keeps the same key set as DIFF32_OPS forever
-- a key in one and not the other silently drops out of every per-op score rather
than failing -- and that each key resolves to the CPU transform class.

Requires the full mmseg/mmengine stack (augmentations.py imports the registries),
so run these in the `sensaug` conda env on a compute node, not on a laptop.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmseg.registry import TRANSFORMS

from sensaug.dataset.augmentations import (
    NON_DIFF32_OPS,
    NON_DIFF32_OPS_GEOMETRIC,
    HSVPerturbation,
    LEGACY20_OPS,
    RandomTrainTransformNew,
    perturb_hsv,
    resolve_perturbation_set,
)
from sensaug.dataset.differentiable_augmentations_aa import (
    DIFF32_OPS,
    GEOMETRIC_OP_KEYS,
)
from sensaug.runner_utils import _perturbation_transform_cfg


@pytest.fixture
def bgr_image():
    return (np.random.default_rng(0).random((13, 19, 3)) * 255).astype(np.uint8)


@pytest.fixture
def seg_map():
    return (np.random.default_rng(1).integers(0, 19, size=(13, 19))).astype(np.uint8)


# --- registry shape -----------------------------------------------------------


def test_covers_exactly_the_ops_r_is_computed_over():
    """Index-compatibility with R's axes is the whole premise. An op present in one
    registry and not the other does not raise -- it silently gets no redundancy
    signal and is left at weight 1 forever. The import-time assertion in
    augmentations.py is the real guard; this is its regression test."""
    assert set(NON_DIFF32_OPS) == set(DIFF32_OPS)
    assert len(NON_DIFF32_OPS) == 32


def test_iteration_order_matches_the_matrix_axis_order():
    """PerturbationSensitivityAnalysisHookWithGradients builds R's axes from
    `list(DIFFERENTIABLE_PERTURBATIONS)`. Anything zipping a per-op score against
    this registry positionally rather than by name has to see the same order."""
    assert list(NON_DIFF32_OPS) == list(DIFF32_OPS)


def test_mirrors_the_other_registries_dict_shape():
    """All three registries are read through `transform_cls, _ = REGISTRY[name]`,
    so they have to be interchangeable at every lookup site."""
    for name, value in NON_DIFF32_OPS.items():
        assert isinstance(value, tuple) and len(value) == 2, name
        transform_cls, flag = value
        assert isinstance(transform_cls, type), name
        assert isinstance(flag, bool), name


def test_resolves_to_a_cpu_transform_class_not_a_gpu_op():
    """The two 32-key sets differ in VALUE SHAPE, not just in value: this one maps
    a name to a `(transform_cls, bool)` pair, DIFF32_OPS maps it to a raw callable
    applied to a batch. Anything that unpacks one as the other is a bug, and
    resolve_perturbation_set's docstring is the contract."""
    for name, (cpu_cls, _) in NON_DIFF32_OPS.items():
        gpu_op = DIFF32_OPS[name]
        assert isinstance(cpu_cls, type), name
        assert not isinstance(gpu_op, type), name
        assert callable(gpu_op), name


def test_the_eighteen_autoaugment_ops_are_the_new_perturbations_classes():
    """These are not reimplementations. `rotate_pos` and `Rotate` are one class
    reached under two names -- which is what makes the correspondence between the
    two vocabularies checkable rather than asserted."""
    shared = {
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
    for diff_name, new_name in shared.items():
        assert NON_DIFF32_OPS[diff_name][0] is LEGACY20_OPS[new_name][0], (
            diff_name
        )


def test_posterize_and_solarize_are_deliberately_absent():
    """They are in LEGACY20_OPS but have no differentiable counterpart, so R
    cannot see them. Including them would put ops in the sampled vocabulary that no
    per-op score can ever reach."""
    present = {cls for cls, _ in NON_DIFF32_OPS.values()}
    assert LEGACY20_OPS["PosterizeTransform"][0] not in present
    assert LEGACY20_OPS["SolarizeTransform"][0] not in present


def test_every_class_is_registered_and_importable_by_name():
    """_perturbation_transform_cfg emits `type=<class name>` for mmengine to build,
    and dataloader workers started with `spawn` pickle the dataset -- which needs
    the class findable by qualified name, not just through the registry dict."""
    import sensaug.dataset.augmentations as aug

    for name, (transform_cls, _) in NON_DIFF32_OPS.items():
        assert TRANSFORMS.get(transform_cls.__name__) is transform_cls, name
        assert getattr(aug, transform_cls.__name__, None) is transform_cls, name


# --- the six new HSV ops ------------------------------------------------------

HSV_OPS = ["lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V"]


@pytest.mark.parametrize(
    "name,channel,direction",
    [
        ("lighter_H", 0, 1),
        ("darker_H", 0, 0),
        ("lighter_S", 1, 1),
        ("darker_S", 1, 0),
        ("lighter_V", 2, 1),
        ("darker_V", 2, 0),
    ],
)
def test_hsv_wrapper_equals_hsv_perturbation(name, channel, direction, bgr_image):
    """The six wrappers add a magnitude-shaped constructor and nothing else. If
    they drifted from HSVPerturbation the legacy SA path and the aligned path would
    be measuring different augmentations under one name."""
    transform_cls, _ = NON_DIFF32_OPS[name]

    through_wrapper = transform_cls(magnitude=0.4)({"img": bgr_image.copy()})["img"]
    direct = HSVPerturbation(channel=channel, alpha=0.4, direction=direction)(
        {"img": bgr_image.copy()}
    )["img"]

    assert np.array_equal(through_wrapper, direct)


def test_darkening_v_rails_to_a_floor_not_to_black():
    """The one channel-specific special case in perturb_hsv. Darkening V to 0 would
    make every image identically black at magnitude 1, so darker_V would be
    indistinguishable from any other op that blacks the image out -- and its column
    of R would be measuring nothing."""
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    darkened = perturb_hsv({"img": img.copy()}, channel=2, alpha=1.0, direction=0)["img"]

    assert darkened.max() > 0, "darker_V at full magnitude collapsed the image to black"
    assert darkened.max() < 200


def test_hue_saturates_at_180_not_255():
    """cv2's 8-bit HSV packs hue into [0, 180]. Using 255 as the ceiling would wrap
    the hue past red and back, making lighter_H non-monotonic in magnitude."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    results = perturb_hsv({"img": img.copy()}, channel=0, alpha=1.0, direction=1)
    assert results["img"].shape == img.shape


def test_hsv_ops_do_not_rewrite_ori_shape(bgr_image):
    """perturb_rgb clobbers ori_shape with the current image shape; perturb_hsv
    deliberately does not, matching HSVPerturbation, whose behaviour the legacy
    non-"_new" SA path depends on."""
    results = {"img": bgr_image.copy(), "ori_shape": (99, 99)}
    transform_cls, _ = NON_DIFF32_OPS["lighter_H"]
    assert transform_cls(magnitude=0.3)(results)["ori_shape"] == (99, 99)


# --- every op behaves like a transform ----------------------------------------


@pytest.mark.parametrize("name", sorted(NON_DIFF32_OPS))
def test_constructs_from_a_magnitude_and_round_trips_an_image(name, bgr_image):
    """RandomTrainTransformNew and _perturbation_transform_cfg both build these as
    `cls(magnitude=level)`. An op taking any other signature would raise inside a
    dataloader worker, where the traceback is close to unreadable."""
    transform_cls, _ = NON_DIFF32_OPS[name]

    out = transform_cls(magnitude=0.4)({"img": bgr_image.copy()})["img"]

    assert out.dtype == np.uint8, name
    assert out.shape == bgr_image.shape, name
    assert np.isfinite(out).all(), name


@pytest.mark.parametrize("name", sorted(NON_DIFF32_OPS))
def test_full_magnitude_actually_changes_pixels(name, bgr_image):
    """Mirrors runner_utils.verify_perturbation_effective. An op that is a no-op at
    magnitude 1 contributes a constant row to D_grad, gets dropped by `correlate`,
    and leaves a NaN row in R -- silently, and only visible in the `dropped` list."""
    transform_cls, _ = NON_DIFF32_OPS[name]

    np.random.seed(0)  # `noise` draws a fresh sample per call
    out = transform_cls(magnitude=1.0)({"img": bgr_image.copy()})["img"]

    assert not np.array_equal(out, bgr_image), f"{name} at magnitude 1.0 is a no-op"


@pytest.mark.parametrize("name", sorted(NON_DIFF32_OPS_GEOMETRIC))
def test_geometric_ops_warp_the_label_with_the_image(name, bgr_image, seg_map):
    """The property that makes these safe to TRAIN on, and the one their
    differentiable counterparts lack: aa.py documents that neither _DiffAugTransform
    nor CollectGradientHook._grad_for_op moves gt_seg_map, so the diff geometrics'
    measured dL/dmagnitude is dominated by image-label misalignment. The CPU classes
    warp both, so training on them is not learning from a misregistered target."""
    transform_cls, _ = NON_DIFF32_OPS[name]

    results = transform_cls(magnitude=0.8)(
        {"img": bgr_image.copy(), "gt_seg_map": seg_map.copy()}
    )

    assert not np.array_equal(results["gt_seg_map"], seg_map), (
        f"{name} moved the image but left the label -- training on this pairs each "
        f"pixel with the wrong class"
    )
    assert results["gt_seg_map"].shape == seg_map.shape


# --- perturbation set selection -----------------------------------------------


def test_resolve_perturbation_set_selects_the_non_diff32_registry():
    assert resolve_perturbation_set("non-diff32") is NON_DIFF32_OPS


def test_geometric_and_photometric_filters_partition_the_non_diff32_set():
    """This set has both kinds of op. Silently returning all 32 under
    --geometric-only would run a differently-scoped experiment than was asked for."""
    geometric = resolve_perturbation_set("aligned", geometric_only=True)
    photometric = resolve_perturbation_set("aligned", photometric_only=True)

    assert set(geometric) == set(GEOMETRIC_OP_KEYS)
    assert len(geometric) == 10
    assert len(photometric) == 22
    assert set(geometric).isdisjoint(photometric)
    assert set(geometric) | set(photometric) == set(NON_DIFF32_OPS)


def test_unknown_perturbation_set_still_raises():
    with pytest.raises(ValueError, match="unknown perturbation_set"):
        resolve_perturbation_set("legacy")


# --- cfg resolution is vocabulary-aware ---------------------------------------


@pytest.mark.parametrize("name", sorted(NON_DIFF32_OPS))
def test_cfg_resolution_honours_the_non_diff32_vocabulary(name):
    """The key is not the registered type name here (`lighter_R` -> `LighterR`),
    so the class name has to come off the registry. Returning the key would name a
    transform that does not exist and trip _assert_transforms_present."""
    expected_cls, _ = NON_DIFF32_OPS[name]
    transform_name, cfg = _perturbation_transform_cfg(
        name, 0.25, perturbation_set="non-diff32"
    )

    assert transform_name == expected_cls.__name__
    assert cfg == {"type": expected_cls.__name__, "magnitude": 0.25}
    assert TRANSFORMS.build(cfg).__class__.__name__ == transform_name


@pytest.mark.parametrize("name", sorted(NON_DIFF32_OPS))
def test_cfg_resolution_without_the_argument_falls_back_to_the_cpu_class(name):
    """With the argument omitted the name-based fallback runs. The Diff* wrappers
    it used to reach are gone, so a 32-key name now lands on the CPU class -- the
    only pipeline transform that still exists under that name."""
    expected_cls, _ = NON_DIFF32_OPS[name]
    assert _perturbation_transform_cfg(name, 0.25)[0] == expected_cls.__name__


def test_gpu_set_has_no_pipeline_transform():
    """The GPU set is applied batched after collation, so asking for a pipeline cfg
    is a caller that forgot to branch. It raises rather than resolving onto the CPU
    class of the same name, which would silently augment differently from what the
    probe measures."""
    with pytest.raises(ValueError, match="no pipeline transforms"):
        _perturbation_transform_cfg("lighter_R", 0.25, perturbation_set="diff32")


def test_legacy20_names_are_unaffected_by_the_argument():
    """LEGACY20_OPS keys are the registered type names themselves, and no
    32-op key collides with one, so passing perturbation_set must not perturb
    that path."""
    for tag in (None, "legacy20", "non-diff32"):
        assert _perturbation_transform_cfg(
            "BrightnessTransform", 0.25, perturbation_set=tag
        ) == ("BrightnessTransform", {"type": "BrightnessTransform", "magnitude": 0.25})


# --- the reweighted pdf survives the real sampler -----------------------------
#
# sensaug/redundancy.py is tested standalone in test_redundancy.py, on synthetic
# input and with no mmseg. These are the few claims that need the actual sampler:
# the pdf it produces has to be one RandomTrainTransformNew will accept and can
# draw from over the non-diff32 vocabulary.


def _aligned_pdf(levels=(0.2, 0.5, 0.8)):
    """A generate_pdf_new-shaped pdf over the aligned vocabulary."""
    names = list(NON_DIFF32_OPS)
    none_mass = 1.0 / (len(names) + 1)
    per_entry = (1.0 - none_mass) / (len(names) * len(levels))
    pdf = {(name, level): per_entry for name in names for level in levels}
    pdf[("none", 0)] = none_mass
    return pdf


@pytest.mark.parametrize("lam", [0.0, 0.25, 1.0])
def test_the_reweighted_pdf_is_accepted_by_the_real_sampler(lam):
    """RandomTrainTransformNew raises if the probabilities do not sum to 1 within
    1e-6, and it raises in its constructor -- which runs inside a dataloader worker,
    where the traceback surfaces as an opaque worker crash rather than as anything
    naming the pdf. Independent of whether R means anything: it exercises the
    normalisation, the key shape and the sampler contract."""
    from sensaug.redundancy import compute_red, reweight

    rng = np.random.default_rng(0)
    n = len(NON_DIFF32_OPS)
    r = rng.normal(0, 0.3, size=(n, n))
    r = (r + r.T) / 2
    np.fill_diagonal(r, 1.0)
    red = compute_red(r, list(NON_DIFF32_OPS)).as_dict()

    pdf = reweight(_aligned_pdf(), red, lam).pdf

    transform = RandomTrainTransformNew(pdf_dict=pdf, perturbation_set="non-diff32")
    assert transform._probs.sum() == pytest.approx(1.0)


def test_the_sampler_draws_the_aligned_vocabulary_and_changes_pixels(bgr_image):
    """End to end: a reweighted pdf, through the real sampler, over the aligned
    registry. Guards the two ways this can be wired up wrong and still look fine --
    a pdf key the registry cannot resolve, and a transform that silently no-ops."""
    from sensaug.redundancy import compute_red, reweight

    rng = np.random.default_rng(1)
    n = len(NON_DIFF32_OPS)
    r = rng.normal(0, 0.3, size=(n, n))
    r = (r + r.T) / 2
    np.fill_diagonal(r, 1.0)
    red = compute_red(r, list(NON_DIFF32_OPS)).as_dict()

    pdf = reweight(_aligned_pdf(levels=(0.9,)), red, 0.5).pdf
    pdf.pop(("none", 0))  # force an augmentation every draw
    total = sum(pdf.values())
    pdf = {k: v / total for k, v in pdf.items()}

    transform = RandomTrainTransformNew(pdf_dict=pdf, perturbation_set="non-diff32")

    np.random.seed(0)
    changed = 0
    for _ in range(20):
        out = transform({"img": bgr_image.copy(), "gt_seg_map": seg_map_for(bgr_image)})
        assert out["img"].dtype == np.uint8
        if not np.array_equal(out["img"], bgr_image):
            changed += 1

    assert changed == 20, f"only {changed}/20 draws altered the image"


def seg_map_for(img):
    return np.zeros(img.shape[:2], dtype=np.uint8)
