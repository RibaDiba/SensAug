"""The --aug-type baselines must warp the label with the image.

torch/torchvision only: imports the label-safe module and the photometric
spaces directly, not sensaug.dataset.augmentations (which pulls in mmseg).
"""

import numpy as np
import pytest
import torch
from torchvision.transforms.v2 import (
    AugMix,
    AutoAugment,
    RandAugment,
    TrivialAugmentWide,
)

from sensaug.dataset.utils.label_safe_transforms import (
    GEOMETRIC_TRANSFORM_IDS,
    SEG_IGNORE_INDEX,
    LabelSafeAutoAugment,
    LabelSafeRandAugment,
    LabelSafeTrivialAugmentWide,
)
from sensaug.dataset.utils.non_geometric_transforms import (
    ColorAugMix,
    ColorAutoAugment,
    ColorRandAugment,
    ColorTrivialAugmentWide,
)

LABEL_SAFE = [
    LabelSafeAutoAugment,
    LabelSafeRandAugment,
    LabelSafeTrivialAugmentWide,
]


def _square(h=96, w=128):
    """An image whose every pixel's class is readable back off its colour."""
    img = np.full((h, w, 3), 40, np.uint8)
    img[24:72, 32:96] = 200
    seg = np.zeros((h, w), np.uint8)
    seg[24:72, 32:96] = 1
    return img, seg


@pytest.mark.parametrize("cls", LABEL_SAFE)
def test_label_follows_image(cls):
    class Spy(cls):
        """Records the ops forward() applied to the image. The mixin replays
        geometric ones on the label by calling past this override, so only
        image ops land here."""

        def _apply_image_or_video_transform(self, image, tid, *args, **kwargs):
            self.ops.append(tid)
            return super()._apply_image_or_video_transform(image, tid, *args, **kwargs)

    torch.manual_seed(0)
    aug = Spy()
    geometric_seen = 0
    for _ in range(300):
        img, seg = _square()
        aug.ops = ops = []
        out, seg_out = aug.apply(img, seg)

        assert out.shape == img.shape and out.dtype == np.uint8
        assert seg_out.shape == seg.shape and seg_out.dtype == seg.dtype
        assert set(np.unique(seg_out)) <= {0, 1, SEG_IGNORE_INDEX}
        if not GEOMETRIC_TRANSFORM_IDS & set(ops):
            np.testing.assert_array_equal(seg_out, seg)
            continue
        geometric_seen += 1
        # Where the label still says square vs background, the image must agree
        # -- judged only where the two classes stayed distinguishable (a
        # photometric op after the warp may have flattened them).
        grey = out.astype(int).mean(-1)
        inside, outside = grey[seg_out == 1], grey[seg_out == 0]
        if inside.size == 0 or outside.size == 0:
            continue
        if abs(np.median(inside) - np.median(outside)) < 20:
            continue
        mid = (np.median(inside) + np.median(outside)) / 2
        pred = (grey > mid) == (np.median(inside) > mid)
        known = seg_out != SEG_IGNORE_INDEX
        agree = (pred[known] == (seg_out[known] == 1)).mean()
        assert agree > 0.97, (cls.__name__, ops, agree)
    assert geometric_seen > 10, "never exercised a geometric op"


@pytest.mark.parametrize("cls", LABEL_SAFE)
def test_out_of_frame_is_ignore(cls):
    aug = cls()
    img, seg = _square()
    aug._seg = torch.from_numpy(seg)[None]
    aug._apply_image_or_video_transform(
        torch.from_numpy(img).permute(2, 0, 1),
        "TranslateX",
        40.0,
        aug.interpolation,
        aug.fill,
    )
    assert (aug._seg[0, :, :40] == SEG_IGNORE_INDEX).all()


def test_size_mismatch_refused():
    img, seg = _square()
    with pytest.raises(ValueError):
        LabelSafeRandAugment().apply(img, seg[:-1])


def test_label_safe_keeps_reference_method():
    """Same op spaces as torchvision -- nothing about the method changed."""
    for safe, ref in [
        (LabelSafeAutoAugment, AutoAugment),
        (LabelSafeRandAugment, RandAugment),
        (LabelSafeTrivialAugmentWide, TrivialAugmentWide),
    ]:
        assert safe._AUGMENTATION_SPACE is ref._AUGMENTATION_SPACE


@pytest.mark.parametrize(
    "mine,ref,n_bins",
    [
        (ColorAutoAugment, AutoAugment, 10),
        (ColorRandAugment, RandAugment, 31),
        (ColorTrivialAugmentWide, TrivialAugmentWide, 31),
        (ColorAugMix, AugMix, 11),
    ],
)
def test_photometric_spaces_match_reference(mine, ref, n_bins):
    """Each photometric variant is the reference table minus the geometric ops.

    Pins the old bug: every variant used AugMix's 4-to-0-bit Posterize, where
    AutoAugment/RandAugment use 8-to-4 and TrivialAugmentWide 8-to-2 -- down to
    a 0-bit (all black) image.
    """
    expected = {k for k in ref._AUGMENTATION_SPACE if k not in GEOMETRIC_TRANSFORM_IDS}
    assert set(mine._AUGMENTATION_SPACE) == expected
    for k in expected:
        a = mine._AUGMENTATION_SPACE[k][0](n_bins, 64, 64)
        b = ref._AUGMENTATION_SPACE[k][0](n_bins, 64, 64)
        assert (a is None and b is None) or torch.equal(a, b), k
