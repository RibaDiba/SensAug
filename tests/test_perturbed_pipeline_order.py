"""Regression tests for the eval pipeline `_build_perturbed_pipeline` hands back.

mmseg's test pipelines load the annotation *after* Resize on purpose, so the
label stays at `ori_shape` -- which is where `EncoderDecoder.postprocess_result`
puts the prediction back. `_build_perturbed_pipeline` moves both transforms to
index 1, so whichever move runs last owns the slot; when LoadAnnotations won it,
the label got resized with the image and `IoUMetric.intersect_and_union` indexed
an `[H, W]` prediction with an `[H', W']` mask:

    IndexError: The shape of the mask [512, 699] at index 0 does not match
                the shape of the indexed tensor [366, 500] at index 0

That killed every perturbed eval on pascal_voc12 and ade20k (test_robust.py, and
the SA round-eval on the same code path). Cityscapes was immune only because its
test scale equals its native image size, making the resize a no-op.

Requires the full mmseg/mmengine stack (runner_utils imports the registries), so
run these in the `sensaug` conda env, not on a laptop.
"""

import os
import sys

import pytest

pytest.importorskip("mmseg")
pytestmark = pytest.mark.requires_mmseg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmengine.config import ConfigDict

from sensaug.runner_utils import _build_perturbed_pipeline


def _eval_cfg():
    """The test_pipeline every config under _base_/datasets/ ships."""
    return ConfigDict(
        dataset=dict(
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="Resize", scale=(2048, 512), keep_ratio=True),
                dict(type="LoadAnnotations"),
                dict(type="PackSegInputs"),
            ]
        )
    )


def _train_cfg():
    return ConfigDict(
        dataset=dict(
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations"),
                dict(type="RandomResize", scale=(2048, 512), ratio_range=(0.5, 2.0)),
                dict(type="RandomCrop", crop_size=(512, 512)),
                dict(type="RandomFlip", prob=0.5),
                dict(type="PackSegInputs"),
            ]
        )
    )


def _types(cfg):
    return [t["type"] for t in cfg.dataset.pipeline]


def test_resize_stays_ahead_of_load_annotations():
    cfg = _eval_cfg()
    inserted = _build_perturbed_pipeline(
        cfg, {"ShearX": 0.25}, train=False, perturbation_set="legacy20"
    )

    assert inserted == ["ShearX"]
    assert _types(cfg) == [
        "LoadImageFromFile",
        "Resize",
        "LoadAnnotations",
        "ShearX",
        "PackSegInputs",
    ]


def test_resize_ahead_of_load_annotations_even_if_cfg_has_them_swapped():
    """The reorder normalizes, rather than assuming the config is already right."""
    cfg = _eval_cfg()
    pipeline = cfg.dataset.pipeline
    pipeline.insert(1, pipeline.pop(2))  # LoadAnnotations before Resize
    assert _types(cfg)[1] == "LoadAnnotations"

    _build_perturbed_pipeline(
        cfg, {"ShearX": 0.25}, train=False, perturbation_set="legacy20"
    )

    types = _types(cfg)
    assert types.index("Resize") < types.index("LoadAnnotations")


def test_perturbation_lands_after_both_loads_and_before_pack():
    cfg = _eval_cfg()
    _build_perturbed_pipeline(
        cfg, {"ShearX": 0.25}, train=False, perturbation_set="legacy20"
    )

    types = _types(cfg)
    assert types.index("LoadAnnotations") < types.index("ShearX")
    assert types[-1] == "PackSegInputs"


def test_clean_reset_leaves_the_pipeline_untouched():
    cfg = _eval_cfg()
    before = _types(cfg)
    assert _build_perturbed_pipeline(cfg, {}, train=False) == []
    assert _types(cfg) == before


def test_train_pipeline_order_is_unchanged():
    """The train path never moves Resize, and its LoadAnnotations is already at
    index 1 -- so the fix is a no-op there and training stays reproducible."""
    cfg = _train_cfg()
    before = _types(cfg)
    _build_perturbed_pipeline(
        cfg, {"ShearX": 0.25}, train=True, perturbation_set="legacy20"
    )

    assert _types(cfg) == before[:-1] + ["ShearX", "PackSegInputs"]
