"""AutoAugment / RandAugment / TrivialAugmentWide that warp the segmentation label.

torchvision's v2 auto-augment classes transform exactly one image and refuse a
``Mask`` input outright (torchvision 0.15), so a segmentation pipeline that hands
them only ``results["img"]`` shears, translates and rotates the image while the
label stays put -- on ~20-60% of samples depending on the method.

These subclasses keep the reference methods unchanged (same op spaces, same
policies, same magnitude tables, same sampling) and only add one thing: every
geometric op the base class applies to the image is replayed on the label with
the identical transform id and magnitude, nearest-neighbour, with out-of-frame
pixels filled with the ignore index. torchvision routes every op of all three
methods through ``_apply_image_or_video_transform``, so that one override is the
whole mechanism.

AugMix is deliberately absent: it blends several independently augmented copies
of the image, so no single warped label matches its output. Use the
photometric-only ``ColorAugMix`` for segmentation instead.

torch/torchvision only -- no mmseg -- so it is testable without the registry.
"""

from typing import Optional

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import AutoAugment, RandAugment, TrivialAugmentWide

GEOMETRIC_TRANSFORM_IDS = frozenset(
    {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}
)

SEG_IGNORE_INDEX = 255


class _LabelSafeMixin:
    """Replays each geometric op on ``self._seg`` while ``forward`` runs."""

    _seg: Optional[torch.Tensor] = None

    def _apply_image_or_video_transform(
        self, image, transform_id, magnitude, interpolation, fill
    ):
        out = super()._apply_image_or_video_transform(
            image, transform_id, magnitude, interpolation, fill
        )
        if self._seg is not None and transform_id in GEOMETRIC_TRANSFORM_IDS:
            self._seg = super()._apply_image_or_video_transform(
                self._seg,
                transform_id,
                magnitude,
                InterpolationMode.NEAREST,
                {type(self._seg): SEG_IGNORE_INDEX},
            )
        return out

    def apply(self, img: np.ndarray, seg: Optional[np.ndarray]):
        """Augment an HWC BGR uint8 image (mmseg's layout) and its HW label.

        Converted to RGB for the call, because torchvision's Color op takes its
        grayscale from RGB luma weights. The label must share the image's
        spatial size -- Translate magnitudes are drawn in the image's pixels.
        """
        rgb = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1)
        if seg is not None:
            if seg.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f"{type(self).__name__}: label {seg.shape[:2]} and image "
                    f"{img.shape[:2]} differ in size; geometric ops would not line up"
                )
            self._seg = torch.from_numpy(np.ascontiguousarray(seg))[None]
        try:
            rgb = self(rgb)
            seg_out = None if seg is None else self._seg[0].numpy().astype(seg.dtype)
        finally:
            self._seg = None
        img_out = np.ascontiguousarray(rgb.permute(1, 2, 0).numpy()[..., ::-1])
        return img_out, seg_out


class LabelSafeAutoAugment(_LabelSafeMixin, AutoAugment):
    pass


class LabelSafeRandAugment(_LabelSafeMixin, RandAugment):
    pass


class LabelSafeTrivialAugmentWide(_LabelSafeMixin, TrivialAugmentWide):
    pass
