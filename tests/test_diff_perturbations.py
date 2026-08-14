"""Tests for the DIFF_PERTURBATIONS pipeline wrappers.

These are what let the sensitivity analysis measure the SAME augmentations the
gradient probe differentiates. Before them the two pipelines ran over vocabularies
with zero overlap (NEW_PERTURBATIONS vs DIFFERENTIABLE_PERTURBATIONS), so no SA
magnitude could ever be looked up for an op appearing in R.

Requires the full mmseg/mmengine stack (augmentations.py imports the registries),
so run these in the `sensaug` conda env on a compute node, not on a laptop.
"""

import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmseg.registry import TRANSFORMS


from sensaug.dataset.augmentations import (
    DIFF_PERTURBATIONS,
    NEW_PERTURBATIONS,
    RandomAlphaTrainTransform,
    RandomTrainTransformNew,
    resolve_perturbation_set,
)
from sensaug.dataset.differentiable_augmentations import (
    DIFFERENTIABLE_PERTURBATIONS,
    img_to_rgb01,
    rgb01_to_img,
)
from sensaug.dataset.differentiable_augmentations_aa import (
    ALL_DIFFERENTIABLE_PERTURBATIONS,
    GEOMETRIC_OP_KEYS,
)
from sensaug.runner_utils import _perturbation_transform_cfg


@pytest.fixture
def bgr_image():
    return (np.random.default_rng(0).random((13, 19, 3)) * 255).astype(np.uint8)


# --- registry shape -----------------------------------------------------------


def test_covers_exactly_the_merged_thirty_two_ops():
    """No op may be added or dropped in the wrapping. A missing one would vanish
    from the SA curve while still appearing in R, and be probed at the fallback
    magnitude forever with nothing in the log to say so.

    32 = the 14 ops ported from the reference repo + the 18 AutoAugment-family
    ops. The base module's DIFFERENTIABLE_PERTURBATIONS still means only the
    former, so this checks against the merged registry the wrappers are
    generated from."""
    assert set(DIFF_PERTURBATIONS) == set(ALL_DIFFERENTIABLE_PERTURBATIONS)
    assert len(DIFF_PERTURBATIONS) == 32
    assert set(DIFFERENTIABLE_PERTURBATIONS) < set(DIFF_PERTURBATIONS)


def test_mirrors_the_new_perturbations_dict_shape():
    """Both registries must be interchangeable at every lookup site, all of which
    do `transform_cls, _ = REGISTRY[name]`."""
    sample = next(iter(NEW_PERTURBATIONS.values()))
    assert isinstance(sample, tuple) and len(sample) == 2

    for name, value in DIFF_PERTURBATIONS.items():
        assert isinstance(value, tuple) and len(value) == 2, name
        transform_cls, flag = value
        assert isinstance(transform_cls, type), name
        assert isinstance(flag, bool), name


def test_every_wrapper_is_registered_under_its_class_name():
    """_perturbation_transform_cfg emits `type=<class name>`; mmengine has to be
    able to build that."""
    for name, (transform_cls, _) in DIFF_PERTURBATIONS.items():
        assert TRANSFORMS.get(transform_cls.__name__) is transform_cls, name


def test_wrapper_classes_are_importable_by_name():
    """Dataloader workers started with `spawn` pickle the dataset, and pickling a
    class requires finding it by qualified name. A class only reachable through the
    registry dict would fail there and nowhere else."""
    import sensaug.dataset.augmentations as aug

    for transform_cls, _ in DIFF_PERTURBATIONS.values():
        assert getattr(aug, transform_cls.__name__, None) is transform_cls


# --- the wrapper is the op ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(DIFFERENTIABLE_PERTURBATIONS))
def test_wrapper_output_equals_the_raw_op(name, bgr_image):
    """THE claim the whole approach rests on: SA scores the same function the probe
    differentiates. If these ever diverge, the SA magnitude handed to the probe
    describes a different augmentation than the one being measured."""
    transform_cls, _ = DIFF_PERTURBATIONS[name]

    torch.manual_seed(0)  # `noise` draws a fresh sample per call
    through_wrapper = transform_cls(magnitude=0.4)({"img": bgr_image.copy()})["img"]

    torch.manual_seed(0)
    raw = DIFFERENTIABLE_PERTURBATIONS[name](img_to_rgb01(bgr_image), 0.4)

    assert np.array_equal(through_wrapper, rgb01_to_img(raw))


def test_wrapper_does_not_mutate_ori_shape(bgr_image):
    """These ops do not resize. Rewriting ori_shape (as Blur/Noise do) would be
    harmless here but is exactly the kind of thing that breaks evaluation if a
    Resize is ever put ahead of them in the pipeline."""
    transform_cls, _ = DIFF_PERTURBATIONS["lighter_R"]
    results = {"img": bgr_image.copy(), "ori_shape": (99, 99)}
    assert transform_cls(magnitude=0.3)(results)["ori_shape"] == (99, 99)


# --- pipeline cfg resolution --------------------------------------------------


@pytest.mark.parametrize("name", sorted(DIFFERENTIABLE_PERTURBATIONS))
def test_perturbation_cfg_resolves_diff_names(name):
    """The key is NOT the registered type name here (`lighter_R` -> `DiffLighterR`),
    unlike NEW_PERTURBATIONS. Returning the key would name a transform that does not
    exist."""
    expected_cls, _ = DIFF_PERTURBATIONS[name]
    transform_name, cfg = _perturbation_transform_cfg(name, 0.25)

    assert transform_name == expected_cls.__name__
    assert cfg == {"type": expected_cls.__name__, "magnitude": 0.25}
    # _assert_transforms_present compares against t.__class__.__name__, so the
    # returned name has to match what building the cfg actually produces.
    assert TRANSFORMS.build(cfg).__class__.__name__ == transform_name


def test_unknown_perturbation_still_raises():
    with pytest.raises(ValueError, match="doesn't match anything"):
        _perturbation_transform_cfg("not_an_augmentation", 0.5)


# --- perturbation set selection ----------------------------------------------


def test_resolve_perturbation_set_selects_registries():
    assert resolve_perturbation_set("new") is NEW_PERTURBATIONS
    assert resolve_perturbation_set("diff") is DIFF_PERTURBATIONS


def test_geometric_and_photometric_filters_partition_the_diff_set():
    """This used to assert that geometric_only RAISED for the diff set, because
    every differentiable op was photometric. The AutoAugment-family rotate/shear/
    translate ops ended that, so the filter now has to return the right subset --
    silently returning all 32 under a geometric-only flag would run a
    differently-scoped experiment than asked for, with nothing to reveal it."""
    geometric = resolve_perturbation_set("diff", geometric_only=True)
    photometric = resolve_perturbation_set("diff", photometric_only=True)

    assert set(geometric) == set(GEOMETRIC_OP_KEYS)
    assert len(geometric) == 10
    assert set(geometric).isdisjoint(photometric)
    assert set(geometric) | set(photometric) == set(DIFF_PERTURBATIONS)

    # Everything ported from the reference repo is photometric.
    assert set(DIFFERENTIABLE_PERTURBATIONS) <= set(photometric)


def test_unknown_perturbation_set_raises():
    with pytest.raises(ValueError, match="unknown perturbation_set"):
        resolve_perturbation_set("legacy")


# --- training transforms honour the vocabulary --------------------------------


def test_random_train_transform_applies_a_diff_op(bgr_image):
    """The pdf's op names come from whichever vocabulary SA ran over; looking them
    up in the wrong registry is a KeyError on the first augmented image."""
    pdf = {("lighter_R", 0.8): 1.0}
    transform = RandomTrainTransformNew(pdf_dict=pdf, perturbation_set="diff")
    out = transform({"img": bgr_image.copy()})["img"]
    assert not np.array_equal(out, bgr_image)


def test_random_train_transform_rejects_a_mismatched_vocabulary(bgr_image):
    pdf = {("lighter_R", 0.8): 1.0}
    transform = RandomTrainTransformNew(pdf_dict=pdf, perturbation_set="new")
    with pytest.raises(KeyError):
        transform({"img": bgr_image.copy()})


def test_random_alpha_transform_validates_eagerly():
    """Validated in __init__ so a bad perturbation_set fails at build time rather
    than on the first sampled image inside a dataloader worker, where the traceback
    is far less legible."""
    with pytest.raises(ValueError, match="unknown perturbation_set"):
        RandomAlphaTrainTransform(perturbation_set="nope")


def test_random_alpha_transform_samples_from_diff(bgr_image):
    transform = RandomAlphaTrainTransform(perturbation_set="diff")
    # "none" is in the sampling population, so try a few draws.
    outs = [transform({"img": bgr_image.copy()})["img"] for _ in range(25)]
    assert any(not np.array_equal(o, bgr_image) for o in outs)
