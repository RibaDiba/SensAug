from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from torchvision.transforms.v2 import (
    AugMix as AugMixBase,
    RandAugment as RandAugmentBase,
    TrivialAugmentWide as TrivialAugmentWideBase,
    AutoAugment as AutoAugmentBase,
)
from torchvision.transforms.v2 import (
    AutoAugmentPolicy,
    functional as F,
    InterpolationMode,
    Transform,
)

# The geometric op ids in torchvision's auto-augment spaces. Every photometric
# space below is torchvision's own table with exactly these removed, so the
# magnitudes (Posterize's bit range especially, which differs per method) stay
# the reference method's rather than a hand-copied table.
_GEOMETRIC_OPS = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})


def _photometric(space):
    return {k: v for k, v in space.items() if k not in _GEOMETRIC_OPS}


class ColorAugMix(AugMixBase):
    """AugMix excluding geometric operations"""

    _PARTIAL_AUGMENTATION_SPACE = _photometric(AugMixBase._PARTIAL_AUGMENTATION_SPACE)
    _AUGMENTATION_SPACE = _photometric(AugMixBase._AUGMENTATION_SPACE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ColorAutoAugment(AutoAugmentBase):
    """AutoAugment excluding geometric operations.

    Not the reference ImageNet policy: the 7 of its 25 sub-policies that contain
    a Rotate or ShearX are dropped whole, photometric half included.
    """

    _AUGMENTATION_SPACE = _photometric(AutoAugmentBase._AUGMENTATION_SPACE)

    def _get_policies(
        self, policy: AutoAugmentPolicy
    ) -> List[
        Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]
    ]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                # (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                # (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                # (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                # (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                # (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                # (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                # (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ColorRandAugment(RandAugmentBase):
    """RandAugment excluding geometric operations"""

    _AUGMENTATION_SPACE = _photometric(RandAugmentBase._AUGMENTATION_SPACE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ColorTrivialAugmentWide(TrivialAugmentWideBase):
    """TrivialAugmentWide excluding geometric operations"""

    _AUGMENTATION_SPACE = _photometric(TrivialAugmentWideBase._AUGMENTATION_SPACE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
