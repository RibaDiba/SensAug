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

# Common augmentation space for color transformations excluding spatial operations.
_COLOR_AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "Posterize": (
        lambda num_bins, height, width: (
            4 - (torch.arange(num_bins) / ((num_bins - 1) / 4))
        )
        .round()
        .int(),
        False,
    ),
    "Solarize": (
        lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
        False,
    ),
    "AutoContrast": (lambda num_bins, height, width: None, False),
    "Equalize": (lambda num_bins, height, width: None, False),
    "Brightness": (
        lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
        True,
    ),
    "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
    "Contrast": (
        lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
        True,
    ),
    "Sharpness": (
        lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
        True,
    ),
}


class ColorAugMix(AugMixBase):
    """AugMix excluding geometric operations"""

    _PARTIAL_AUGMENTATION_SPACE = {
        "Posterize": (
            lambda num_bins, height, width: (
                4 - (torch.arange(num_bins) / ((num_bins - 1) / 4))
            )
            .round()
            .int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    _AUGMENTATION_SPACE = {
        **_PARTIAL_AUGMENTATION_SPACE,
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ColorAutoAugment(AutoAugmentBase):
    """AutooAugment excluding geometric operations"""

    _AUGMENTATION_SPACE = {
        "Posterize": (
            lambda num_bins, height, width: (
                4 - (torch.arange(num_bins) / ((num_bins - 1) / 4))
            )
            .round()
            .int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins),
            True,
        ),
        "Invert": (lambda num_bins, height, width: None, False),
    }

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

    _AUGMENTATION_SPACE = _COLOR_AUGMENTATION_SPACE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ColorTrivialAugmentWide(TrivialAugmentWideBase):
    """TrivialAugmentWide excluding geometric operations"""

    _AUGMENTATION_SPACE = _COLOR_AUGMENTATION_SPACE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
