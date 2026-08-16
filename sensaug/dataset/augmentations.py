"""MMSegmentation transformations."""

import numpy as np
import cv2
from PIL import Image

from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS

import torch
from torchvision.transforms.v2 import (
    AugMix,
    RandAugment,
    TrivialAugmentWide,
    AutoAugment,
)
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F
from torchvision.transforms import InterpolationMode

from sensaug.dataset.utils.non_geometric_transforms import (
    ColorAugMix,
    ColorAutoAugment,
    ColorRandAugment,
    ColorTrivialAugmentWide,
)

from sensaug.dataset.utils.cropping import *

# The merged 32-op GPU vocabulary. Imported here only so NON_DIFF32_OPS can be
# checked against it at import time -- the ops themselves are applied by
# sensaug.dataset.gpu_augment, never through this module's pipeline transforms.
from sensaug.dataset.differentiable_augmentations_aa import (
    GEOMETRIC_OP_KEYS,
    DIFF32_OPS as DIFF32_OPS,
)
from sensaug.dataset.parameters import COMBINATION_PARAMETERS
from sensaug.dataset.imagenet_c import (
    motion_blur,
    zoom_blur,
    pixelate,
    jpeg_compression,
    snow,
    frost,
    fog,
)

#: Every valid `perturbation_set` value. See resolve_perturbation_set.
PERTURBATION_SETS = ("legacy20", "non-diff32", "diff32")

#: The one set whose ops are applied batched on GPU (sensaug.dataset.gpu_augment)
#: rather than as pipeline transforms. Anything that builds a dataloader pipeline
#: has to branch on this.
GPU_PERTURBATION_SET = "diff32"


def _reject_gpu_set(name: str, where: str) -> None:
    """Fail loudly when a GPU-set name reaches a CPU pipeline transform.

    Silence here is the specific failure this refactor removed: the GPU set and
    the CPU 32-op set share all 32 keys, so a name-based lookup resolves either
    one without complaint and the only symptom is a run that is quietly 40-150x
    slower per image, or -- worse -- one arm augmenting differently from another.
    """
    if name == GPU_PERTURBATION_SET:
        raise ValueError(
            f"{where} was given perturbation_set={name!r}, which is applied "
            "batched on GPU by sensaug.dataset.gpu_augment and has no pipeline "
            "transform. This transform should not have been inserted at all; see "
            "runner_utils' GPU dispatch."
        )


MAX_COLOR = 1.0
MAX_BLUR = 49
MAX_NOISE = 50.0
MAX_DISTORT = 100.0

PERTURBATIONS = {
    "lighter_R": MAX_COLOR,
    "lighter_G": MAX_COLOR,
    "lighter_B": MAX_COLOR,
    "darker_R": MAX_COLOR,
    "darker_G": MAX_COLOR,
    "darker_B": MAX_COLOR,
    "lighter_H": MAX_COLOR,
    "lighter_S": MAX_COLOR,
    "lighter_V": MAX_COLOR,
    "darker_H": MAX_COLOR,
    "darker_S": MAX_COLOR,
    "darker_V": MAX_COLOR,
    "blur": MAX_BLUR,
    "noise": MAX_NOISE,
}

GEOMETRIC_TRANSFORMS = [
    "ShearX",
    "NegativeShearX",
    "ShearY",
    "NegativeShearY",
    "TranslateX",
    "NegativeTranslateX",
    "TranslateY",
    "NegativeTranslateYRotate",
    "NegativeRotate",
]

IMAGENETC_NAME_FN_DICT = {
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
    "snow": snow,
    "frost": frost,
    "fog": fog,
}


def get_rot_matrix(degrees):
    rad = np.radians(degrees)
    return np.array(
        [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    ).astype(np.float32)


def get_translation_matrix(x=0, y=0):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]]).astype(np.float32)


def get_shear_matrix(x=0, y=0):
    return np.array([[1, x, 0], [y, 1, 0], [0, 0, 1]]).astype(np.float32)


def get_shear_about_center_cv2(img_shape: tuple, x: float = 0, y: float = 0):
    """Assumes use with cv2 warpAffine -- x and y positions are flipped."""
    return (
        get_translation_matrix(x=img_shape[1] // 2, y=img_shape[0] // 2)
        @ get_shear_matrix(x=x, y=y)
        @ get_translation_matrix(x=-img_shape[1] // 2, y=-img_shape[0] // 2)
    )


def get_rot_about_center_cv2(img_shape: tuple, degrees: float):
    """Assumes use with cv2 warpAffine -- x and y positions are flipped."""
    return (
        get_translation_matrix(x=img_shape[1] // 2, y=img_shape[0] // 2)
        @ get_rot_matrix(degrees=degrees)
        @ get_translation_matrix(x=-img_shape[1] // 2, y=-img_shape[0] // 2)
    )


@TRANSFORMS.register_module()
class FastRGBTransform(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, channel: int = 0, alpha: float = 0):
        assert channel == 0 or channel == 1 or channel == 2, (
            "channel must be out of [0, 1, 2]"
        )
        assert alpha >= -1 and alpha <= 1, (
            "alpha must be any value from -1 to 1 inclusive"
        )

        self.channel = channel  # 0 --> B, 1 --> G, 2 --> R
        self.alpha = alpha  # -1 <= alpha <= 1. degree of intensity
        self.direction = int(alpha > 0)  # 0 --> darker, 1 --> lighter

    def transform(self, results: dict) -> dict:
        """Modify the red channel of an image by a certain factor.
        Negative values are in the lighter direction and positive values are in the darker direction.
        If the image is torch Tensor, it is expected
        to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions,

        Args:
            img (PIL Image or Tensor): Image to be modified.

        Returns:
            PIL Image or Tensor: Image with modified red channel
        """

        img = results["img"]
        img[..., self.channel] = (
            img[..., self.channel]
            - self.alpha * img[..., self.channel]
            + 255 * self.direction * self.alpha
        )

        results["img"] = img.astype(np.uint8)

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channel={self.channel}, alpha={self.alpha})"


@TRANSFORMS.register_module()
class FastBlurTransform(BaseTransform):
    """Applies Gaussian blur perturbation assuming sigma in both X and Y directions are equal."""

    def __init__(self, sigma, kernel_size=67):
        self.kernel_size = kernel_size
        self.sigma = sigma

        assert len(sigma) == 2, "sigma should be a tuple of 2"

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = cv2.GaussianBlur(
            img, (self.kernel_size, self.kernel_size), self.sigma[0], self.sigma[1]
        )

        results["img"] = img
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"


@TRANSFORMS.register_module()
class FastNoiseTransform(BaseTransform):
    """Applies Gaussian Noise perturbation to the image."""

    def __init__(self, sigma=0):
        self.sigma = sigma
        self.mean = 0

    def transform(self, results: dict) -> dict:
        img = results["img"]

        noise = np.random.normal(self.mean, self.sigma, size=img.shape) * 255
        noise = noise.astype(np.uint8)
        img = img + noise

        results["img"] = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


####### Label-modifying Augmentations #######


@TRANSFORMS.register_module()
class AugMixTransform(BaseTransform):
    def __init__(self):
        self.augmix = AugMix(
            severity=3,
            mixture_width=3,
            chain_depth=-1,
            alpha=1.0,
            all_ops=True,
            interpolation=InterpolationMode.BILINEAR,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = self.augmix(img_pil)
        # img_np, cropped_rect = crop_and_resize_data(np.array(img_pil).astype(np.uint8))
        results["img"] = img_pil.permute(1, 2, 0).numpy()
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class AutoAugmentTransform(BaseTransform):
    def __init__(self):
        self.autoAugment = AutoAugment(
            policy=AutoAugmentPolicy.IMAGENET,
            interpolation=InterpolationMode.NEAREST,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = self.autoAugment(img_pil)
        # img_np, cropped_rect = crop_and_resize_data(np.array(img_pil).astype(np.uint8))
        results["img"] = img_pil.permute(1, 2, 0).numpy()
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class RandAugmentTransform(BaseTransform):
    def __init__(self):
        self.randAugment = RandAugment(
            num_ops=2,
            magnitude=9,
            num_magnitude_bins=31,
            interpolation=InterpolationMode.NEAREST,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = self.randAugment(img_pil)
        results["img"] = img_pil.permute(1, 2, 0).numpy()
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class TrivialAugmentWideTransform(BaseTransform):
    def __init__(self):
        self.trivialAugment = TrivialAugmentWide(
            num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None
        )

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = self.trivialAugment(img_pil)
        results["img"] = img_pil.permute(1, 2, 0).numpy()
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


####### Photometric AutoAugment Based Augmentations #######


@TRANSFORMS.register_module()
class PhotometricAugMixTransform(BaseTransform):
    def __init__(self):
        self.augmix = ColorAugMix(
            severity=3,
            mixture_width=3,
            chain_depth=-1,
            alpha=1.0,
            all_ops=True,
            interpolation=InterpolationMode.BILINEAR,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = Image.fromarray(img)
        img = self.augmix(img)
        results["img"] = np.array(img).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class PhotometricAutoAugmentTransform(BaseTransform):
    def __init__(self):
        self.autoaugment = ColorAutoAugment(
            policy=AutoAugmentPolicy.IMAGENET,
            interpolation=InterpolationMode.NEAREST,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = Image.fromarray(img)
        img = self.autoaugment(img)
        results["img"] = np.array(img).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class PhotometricRandAugmentTransform(BaseTransform):
    def __init__(self):
        self.color_randaugment = ColorRandAugment(
            num_ops=2,
            magnitude=9,
            num_magnitude_bins=31,
            interpolation=InterpolationMode.NEAREST,
            fill=None,
        )

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = Image.fromarray(img)
        img = self.color_randaugment(img)
        results["img"] = np.array(img).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@TRANSFORMS.register_module()
class PhotometricTrivialAugmentWideTransform(BaseTransform):
    def __init__(self):
        self.color_trivialaugment = ColorTrivialAugmentWide(
            num_magnitude_bins=31, interpolation=InterpolationMode.NEAREST, fill=None
        )

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = Image.fromarray(img)
        img = self.color_trivialaugment(img)
        results["img"] = np.array(img).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


####################


####### Geometric Perturbations #######


@TRANSFORMS.register_module()
class CropResizeTransform(BaseTransform):
    def transform(self, results: dict) -> dict:
        input_data = results["img"]

        if isinstance(input_data, Image.Image):
            input_data = np.array(input_data)

        original_h, original_w = input_data.shape[:2]
        aspect_ratio = original_w / original_h
        cropped_rect, valid_bounds = bound_image(input_data, aspect_ratio)

        if valid_bounds:
            y, x, h, w = cropped_rect
            cropped_data = input_data[y : y + h, x : x + w]
            interpolation = (
                cv2.INTER_NEAREST if len(input_data.shape) == 2 else cv2.INTER_LINEAR
            )
            input_data = cv2.resize(
                cropped_data, (original_w, original_h), interpolation=interpolation
            )

            if results.get("gt_seg_map", None):
                results = apply_cropping_to_labels(results, cropped_rect)

        results["img"] = np.ascontiguousarray(input_data)
        return results


@TRANSFORMS.register_module()
class ShearX(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1]

    def _shear(self, input, fill=0):
        shear_factor = self.magnitude * 0.3
        M = get_shear_about_center_cv2(input.shape, x=shear_factor, y=0)
        sheared_input = cv2.warpAffine(
            input,
            M[:2],
            (input.shape[1], input.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )  # type: ignore

        return sheared_input

    def _shear_img(self, results: dict):
        if results.get("img", None) is not None:
            # orig_h, orig_w = results["img"].shape[:2]
            img = self._shear(results["img"])
            results["img"] = np.ascontiguousarray(img)
            # new_img, cropped_rect = crop_and_resize_data(
            #     img, original_h=orig_h, original_w=orig_w
            # )
            # new_h, new_w = new_img.shape[:2]
            # h, w = results["ori_shape"][:2]
            # w_scale = new_w / w
            # h_scale = new_h / h
            # results["img"] = np.ascontiguousarray(new_img)
            # results["img_shape"] = results["img"].shape[:2]
            # results["pad_shape"] = results["img"].shape[:2]
            # results["ori_shape"] = results["img"].shape[:2]  # necessary
            # results["scale"] = results["img"].shape[:2]
            # results["scale_factor"] = (w_scale, h_scale)
            return results, None
        else:
            return results, None

    def _shear_seg(self, results: dict, bbox=None) -> None:
        """Resize semantic segmentation map."""
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = self._shear(results["gt_seg_map"], fill=255)
            if bbox is not None:
                results = apply_cropping_to_labels(results, bbox)
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"])

        return results

    def transform(self, results: dict) -> dict:
        results, bbox = self._shear_img(results)
        results = self._shear_seg(results, bbox)
        return results


@TRANSFORMS.register_module()
class NegativeShearX(ShearX):
    def __init__(self, magnitude: float):
        self.magnitude = -1.0 * abs(magnitude)


@TRANSFORMS.register_module()
class ShearY(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] # Shearing factor

    def _shear(self, input, fill=0):
        shear_factor = self.magnitude * 0.3
        M = get_shear_about_center_cv2(input.shape, x=0, y=shear_factor)
        sheared_input = cv2.warpAffine(
            input,
            M[:2],
            (input.shape[1], input.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )  # type: ignore

        return sheared_input

    def _shear_img(self, results: dict):
        if results.get("img", None) is not None:
            # orig_h, orig_w = results["img"].shape[:2]
            img = self._shear(results["img"])
            results["img"] = np.ascontiguousarray(img)
            # new_img, cropped_rect = crop_and_resize_data(
            #     img, original_h=orig_h, original_w=orig_w
            # )
            # new_h, new_w = new_img.shape[:2]
            # h, w = results["ori_shape"][:2]
            # w_scale = new_w / w
            # h_scale = new_h / h
            # results["img"] = np.ascontiguousarray(new_img)
            # results["img_shape"] = results["img"].shape[:2]
            # results["pad_shape"] = results["img"].shape[:2]
            # results["ori_shape"] = results["img"].shape[:2]
            # results["scale"] = results["img"].shape[:2]
            # results["scale_factor"] = (w_scale, h_scale)
            return results, None
        else:
            return results, None

    def _shear_seg(self, results: dict, bbox=None) -> None:
        """Resize semantic segmentation map."""
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = self._shear(results["gt_seg_map"], fill=255)
            if bbox is not None:
                results = apply_cropping_to_labels(results, bbox)
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"])

        return results

    def transform(self, results: dict) -> dict:
        results, bbox = self._shear_img(results)
        results = self._shear_seg(results, bbox)

        return results


@TRANSFORMS.register_module()
class NegativeShearY(ShearY):
    def __init__(self, magnitude: float):
        self.magnitude = -1.0 * abs(magnitude)


@TRANSFORMS.register_module()
class TranslateX(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] # Translation factor

    def _translate(self, input, fill=0):
        tx = self.magnitude * (input.shape[1] / 4.0)
        M = np.array([[1, 0, tx], [0, 1, 0]])
        new_input = cv2.warpAffine(
            input,
            M,
            (input.shape[1], input.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )  # type: ignore

        return new_input

    def _translate_img(self, results: dict):
        if results.get("img", None) is not None:
            # orig_h, orig_w = results["img"].shape[:2]
            img = self._translate(results["img"])
            results["img"] = np.ascontiguousarray(img)
            # new_img, cropped_rect = crop_and_resize_data(
            #     img, original_h=orig_h, original_w=orig_w
            # )
            # new_h, new_w = new_img.shape[:2]
            # h, w = results["ori_shape"][:2]
            # w_scale = new_w / w
            # h_scale = new_h / h
            # results["img"] = np.ascontiguousarray(new_img)
            # results["img_shape"] = results["img"].shape[:2]
            # results["pad_shape"] = results["img"].shape[:2]
            # results["ori_shape"] = results["img"].shape[:2]
            # results["scale"] = results["img"].shape[:2]
            # results["scale_factor"] = (w_scale, h_scale)
            return results, None
        else:
            return results, None

    def _translate_seg(self, results: dict, bbox=None) -> None:
        """Resize semantic segmentation map."""
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = self._translate(results["gt_seg_map"], fill=255)
            if bbox is not None:
                results = apply_cropping_to_labels(results, bbox)
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"])

        return results

    def transform(self, results: dict) -> dict:
        results, bbox = self._translate_img(results)
        results = self._translate_seg(results, bbox)

        return results


@TRANSFORMS.register_module()
class NegativeTranslateX(TranslateX):
    def __init__(self, magnitude: float):
        self.magnitude = -1.0 * abs(magnitude)


@TRANSFORMS.register_module()
class TranslateY(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [-1, 1] # Translation factor

    def _translate(self, input, fill=0):
        ty = self.magnitude * (input.shape[0] / 4.0)
        M = np.array([[1, 0, 0], [0, 1, ty]])
        new_input = cv2.warpAffine(
            input,
            M,
            (input.shape[1], input.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )  # type: ignore

        return new_input

    def _translate_img(self, results: dict):
        if results.get("img", None) is not None:
            # orig_h, orig_w = results["img"].shape[:2]
            img = self._translate(results["img"])
            results["img"] = np.ascontiguousarray(img)
            # new_img, cropped_rect = crop_and_resize_data(
            #     img, original_h=orig_h, original_w=orig_w
            # )
            # new_h, new_w = new_img.shape[:2]
            # h, w = results["ori_shape"][:2]
            # w_scale = new_w / w
            # h_scale = new_h / h
            # results["img"] = np.ascontiguousarray(new_img)
            # results["img_shape"] = results["img"].shape[:2]
            # results["pad_shape"] = results["img"].shape[:2]
            # results["ori_shape"] = results["img"].shape[:2]
            # results["scale"] = results["img"].shape[:2]
            # results["scale_factor"] = (w_scale, h_scale)
            return results, None
        else:
            return results, None

    def _translate_seg(self, results: dict, bbox=None) -> None:
        """Resize semantic segmentation map."""
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = self._translate(results["gt_seg_map"], fill=255)
            if bbox is not None:
                results = apply_cropping_to_labels(results, bbox)
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"])

        return results

    def transform(self, results: dict) -> dict:
        results, bbox = self._translate_img(results)
        results = self._translate_seg(results, bbox)

        return results


@TRANSFORMS.register_module()
class NegativeTranslateY(TranslateY):
    def __init__(self, magnitude: float):
        self.magnitude = -1.0 * abs(magnitude)


@TRANSFORMS.register_module()
class Rotate(BaseTransform):
    """Applies rotation to an image and crops the image to remove padding."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] # Rotation angle
        self.max_angle = 45

    def _rotate(self, input, fill=0):
        input = input.copy()
        angle = self.magnitude * self.max_angle
        M = get_rot_about_center_cv2(input.shape, degrees=angle)
        rotated_img = cv2.warpAffine(
            input,
            M[:2],
            (input.shape[1], input.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )  # type: ignore

        return rotated_img

    def _rotate_img(self, results: dict):
        if results.get("img", None) is not None:
            # orig_h, orig_w = results["img"].shape[:2]
            img = self._rotate(results["img"])
            results["img"] = np.ascontiguousarray(img)
            # new_img, cropped_rect = crop_and_resize_data(
            #     img, original_h=orig_h, original_w=orig_w
            # )
            # new_h, new_w = new_img.shape[:2]
            # h, w = results["ori_shape"][:2]
            # w_scale = new_w / w
            # h_scale = new_h / h
            # results["img"] = np.ascontiguousarray(new_img)
            # results["img_shape"] = results["img"].shape[:2]
            # results["pad_shape"] = results["img"].shape[:2]
            # results["ori_shape"] = results["img"].shape[:2]
            # results["scale"] = results["img"].shape[:2]
            # results["scale_factor"] = (w_scale, h_scale)
            return results, None
        else:
            return results, None

    def _rotate_seg(self, results: dict, bbox=None) -> dict:
        """Resize semantic segmentation map."""
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = self._rotate(results["gt_seg_map"], fill=255)
            if bbox is not None:
                results = apply_cropping_to_labels(results, bbox)
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"])
        return results

    def transform(self, results: dict) -> dict:
        results, bbox = self._rotate_img(results)
        results = self._rotate_seg(results, bbox)

        return results


@TRANSFORMS.register_module()
class NegativeRotate(Rotate):
    def __init__(self, magnitude: float):
        super().__init__(magnitude=magnitude)
        self.magnitude = -1.0 * abs(magnitude)


###################


###### Photometric Perturbations #######


@TRANSFORMS.register_module()
class BrightnessTransform(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] --> [1.0, 2.0]

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
        img_pil = F.adjust_brightness(img_pil, 1.0 + self.magnitude)
        img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
        results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)
        results["img"] = np.ascontiguousarray(results["img"])
        results["ori_shape"] = results["img"].shape[:2]  # necessary
        return results


@TRANSFORMS.register_module()
class NegativeBrightnessTransform(BrightnessTransform):
    def __init__(self, magnitude: float):
        self.magnitude = -0.6 * abs(magnitude)  # [0, 1] --> [0.4, 1.0]


@TRANSFORMS.register_module()
class ColorTransform(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] --> [1.0, 3.0]

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
        img_pil = F.adjust_saturation(
            img_pil, 1.0 + self.magnitude * 2.0
        )  # NOTE: scaling magnitude by 2 here
        img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
        results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)
        results["img"] = np.ascontiguousarray(results["img"])
        results["ori_shape"] = results["img"].shape[:2]  # necessary
        return results


@TRANSFORMS.register_module()
class NegativeColorTransform(ColorTransform):
    def __init__(self, magnitude: float):
        self.magnitude = -0.3 * abs(magnitude)  # [0, 1] --> [0.4, 1.0]


@TRANSFORMS.register_module()
class ContrastTransform(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = magnitude  # [0, 1] --> [1.0, 2.0]

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
        img_pil = F.adjust_contrast(img_pil, 1.0 + self.magnitude)
        img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
        results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)
        results["img"] = np.ascontiguousarray(results["img"])
        results["ori_shape"] = results["img"].shape[:2]  # necessary
        return results


@TRANSFORMS.register_module()
class NegativeContrastTransform(ContrastTransform):
    def __init__(self, magnitude: float):
        self.magnitude = -0.6 * abs(magnitude)  # [0, 1] --> [0.4, 1.0]


@TRANSFORMS.register_module()
class SharpnessTransform(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)  # [0, 1] --> [1.0, 2.0]

    def transform(self, results: dict) -> dict:
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
        img_pil = F.adjust_sharpness(img_pil, 1.0 + self.magnitude)
        img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
        results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)
        results["ori_shape"] = results["img"].shape[:2]  # necessary
        return results


@TRANSFORMS.register_module()
class NegativeSharpnessTransform(SharpnessTransform):
    def __init__(self, magnitude: float):
        self.magnitude = -0.6 * abs(magnitude)  # [0, 1] --> [0.4, 1.0]


@TRANSFORMS.register_module()
class PosterizeTransform(BaseTransform):
    def __init__(self, magnitude: int):
        self.magnitude = abs(magnitude)  # [0, 1]

    def transform(self, results: dict) -> dict:
        bits = int(
            (1.0 - self.magnitude) * 4
        )  # [0, 4], bits=8 would make an image entirely black
        img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
        img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
        img_pil = F.posterize(img_pil, bits)
        img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
        results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)
        results["ori_shape"] = results["img"].shape[:2]  # necessary
        return results


@TRANSFORMS.register_module()
class SolarizeTransform(BaseTransform):
    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        if self.magnitude > 0:
            threshold = 255 - int(
                self.magnitude * 255
            )  # when magnitude is 0, threshold won't affect any pixels
            img_pil = torch.tensor(results["img"]).permute(2, 0, 1)
            img_pil = img_pil.flip(dims=(0,))  # BGR -> RGB
            img_pil = F.solarize(img_pil, threshold)
            img_pil = img_pil.flip(dims=(0,))  # RGB -> BGR
            results["img"] = img_pil.permute(1, 2, 0).numpy().astype(np.uint8)

        results["ori_shape"] = results["img"].shape[:2]  # necessary

        return results


###################


@TRANSFORMS.register_module()
class RandomAlphaTrainTransform(BaseTransform):
    """
    Randomly chooses both augmentation class (uniform) and augmentation intensity (uniform).
    """

    def __init__(
        self,
        geometric_only: bool = False,
        photometric_only: bool = False,
        perturbation_set: str = "legacy20",
    ):
        self.geometric_only = geometric_only
        self.photometric_only = photometric_only
        # One of the CPU sets -- "legacy20" or "non-diff32". A string rather than
        # the dict itself because this goes through an mmengine Config, which
        # cannot serialize classes. Validated eagerly so a bad value fails at build
        # time, not on the first sampled image inside a dataloader worker.
        #
        # Never "diff32": those ops are applied batched on GPU by
        # sensaug.dataset.gpu_augment, and this transform is never inserted into
        # the pipeline for that set. The guard below is what says so out loud.
        _reject_gpu_set(perturbation_set, type(self).__name__)
        self.perturbation_set = perturbation_set
        resolve_perturbation_set(perturbation_set, geometric_only, photometric_only)

    def transform(self, results: dict) -> dict:
        results["img"] = np.ascontiguousarray(results["img"].copy())

        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = np.ascontiguousarray(results["gt_seg_map"].copy())

        num_transforms = 1

        perturbations = resolve_perturbation_set(
            self.perturbation_set, self.geometric_only, self.photometric_only
        )
        perturbation_list = list(perturbations.keys()) + ["none"]

        for _ in range(num_transforms):
            # sample perturbation
            perturbation_type = np.random.choice(perturbation_list)
            perturbation_level = np.random.uniform(low=0, high=1, size=None)

            if perturbation_type != "none":
                transform_cls, _ = perturbations[perturbation_type]
                transform = transform_cls(magnitude=perturbation_level)
                results = transform(results)

        return results


@TRANSFORMS.register_module()
class RandomTrainTransformNew(BaseTransform):
    """
    Randomly selects a perturbation from one of the CPU registries, weighted by
    the SA-derived pdf. Then, uses a beta distribution to mix the perturbed image
    with the original image.
    """

    def __init__(self, pdf_dict: dict, perturbation_set: str = "legacy20"):
        self.pdf_dict: dict = pdf_dict
        # Which CPU registry the pdf's op names are drawn from. "legacy20" and
        # "non-diff32" share no names, so a mismatch is a KeyError on the first
        # augmented image rather than anything subtle. See
        # resolve_perturbation_set; "diff32" never reaches here.
        _reject_gpu_set(perturbation_set, type(self).__name__)
        self.perturbation_set = perturbation_set
        self._perturbations = resolve_perturbation_set(perturbation_set)

        # keys are (perturbation, level) pairs, so they cannot be passed to
        # np.random.choice directly -- it only accepts 1-D populations. Sample an
        # index into the key list instead.
        self._keys = list(pdf_dict.keys())
        self._probs = np.asarray(list(pdf_dict.values()), dtype=np.float64)

        total = self._probs.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"RandomTrainTransformNew pdf_dict probabilities sum to {total}, not 1.0"
            )
        self._probs /= total

    def transform(self, results: dict) -> dict:
        num_transforms = 1

        for _ in range(num_transforms):
            # sample perturbation
            perturbation, level = self._keys[
                np.random.choice(len(self._keys), size=None, p=self._probs)
            ]

            if perturbation != "none":
                # generate some gaussian noise for level
                level = np.clip(
                    level + np.random.normal(0, scale=0.1), a_min=0, a_max=1
                )

                transform_cls, _ = self._perturbations[perturbation]
                transform = transform_cls(magnitude=level)
                results = transform(results)

        return results


@TRANSFORMS.register_module()
class ImageNetCTransform(BaseTransform):
    def __init__(self, name, severity=1):
        assert severity <= 5 and severity >= 1, (
            "severity must be in range [1, 5] inclusive"
        )

        self.name = name
        self.severity = severity
        self.transform_fn = IMAGENETC_NAME_FN_DICT[name]

    def transform(self, results: dict) -> dict:
        """Modify the red channel of an image by a certain factor.
        Negative values are in the lighter direction and positive values are in the darker direction.
        If the image is torch Tensor, it is expected
        to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions,

        Args:
            img (PIL Image or Tensor): Image to be modified.

        Returns:
            PIL Image or Tensor: Image with modified red channel
        """

        img = results["img"]
        img = img[..., ::-1]  # BGR -> RGB
        img = Image.fromarray(np.uint8(img)).convert("RGB")
        img = self.transform_fn(img, severity=self.severity)
        img = np.array(img).astype(np.uint8)
        img = img[..., ::-1]  # RGB -> BGR
        results["img"] = img
        results["ori_shape"] = img.shape[:2]

        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, severity={self.severity})"


@TRANSFORMS.register_module()
class RGBPerturbation(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, channel: int = 0, alpha: float = 0, direction: int = 0):
        assert channel == 0 or channel == 1 or channel == 2, (
            "channel must be out of [0, 1, 2]"
        )
        assert alpha >= 0 and alpha <= 1, "alpha must be any value from 0 to 1"
        assert direction == 0 or direction == 1, (
            "direction must be either 0 (darker) or 1 (lighter)"
        )

        self.channel = channel  # 0 --> B, 1 --> G, 2 --> R
        self.alpha = alpha  # 0 <= alpha <= 1. degree of intensity
        self.direction = direction  # 0 --> darker, 1 --> lighter

    def transform(self, results: dict) -> dict:
        """Modify the red channel of an image by a certain factor.
        Negative values are in the lighter direction and positive values are in the darker direction.
        If the image is torch Tensor, it is expected
        to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions,

        Args:
            img (PIL Image or Tensor): Image to be modified.

        Returns:
            PIL Image or Tensor: Image with modified red channel
        """

        img = results["img"]
        img[..., self.channel] = (
            img[..., self.channel]
            - self.alpha * img[..., self.channel]
            + 255 * self.direction * self.alpha
        )

        results["img"] = img.astype(np.uint8)

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(channel={self.channel}, "
            f"alpha={self.alpha}, direction={self.direction})"
        )


@TRANSFORMS.register_module()
class HSVPerturbation(BaseTransform):
    """Applies HSV perturbation to an image.
    Assumes that the image is in RGB format."""

    def __init__(self, channel: int = 0, alpha: float = 0, direction: int = 0):
        assert channel == 0 or channel == 1 or channel == 2, (
            "channel must be out of [0, 1, 2]"
        )
        assert alpha >= 0 and alpha <= 1, "alpha must be any value from 0 to 1"
        assert direction == 0 or direction == 1, (
            "direction must be either 0 (darker) or 1 (lighter)"
        )

        self.channel = channel  # 0 --> R, 1 --> G, 2 --> B
        self.alpha = alpha  # 0 <= alpha <= 1. degree of intensity
        self.direction = direction  # 0 --> darker, 1 --> lighter

        self.rgb2hsv = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        self.hsv2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    def transform(self, results: dict) -> dict:
        """Modify the red channel of an image by a certain factor.
        Negative values are in the lighter direction and positive values are in the darker direction.
        If the image is torch Tensor, it is expected
        to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions,

        Args:
            img (PIL Image or Tensor): Image to be modified.

        Returns:
            PIL Image or Tensor: Image with modified red channel
        """
        return perturb_hsv(
            results, channel=self.channel, alpha=self.alpha, direction=self.direction
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(channel={self.channel}, "
            f"alpha={self.alpha}, direction={self.direction})"
        )


@TRANSFORMS.register_module()
class GaussianBlurPerturbation(BaseTransform):
    """Applies Gaussian blur perturbation assuming sigma in both X and Y directions are equal."""

    def __init__(self, kernel_size=7):
        self.kernel_size = kernel_size

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

        results["img"] = img
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"


@TRANSFORMS.register_module()
class GaussianNoisePerturbation(BaseTransform):
    """Applies Gaussian Noise perturbation to the image."""

    def __init__(self, sigma=0):
        self.sigma = sigma
        self.mean = 0

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = img + np.random.normal(self.mean, self.sigma, size=img.shape)

        results["img"] = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


@TRANSFORMS.register_module()
class RadialDistortionPerturbation(BaseTransform):
    """Applies distortion to both the image and its corresponding annotations."""

    def __init__(self, level=0):
        self.level = level

    @staticmethod
    def distort(img: np.ndarray, level: float) -> np.ndarray:
        """Radial distortion of the image.

        Args:
            img: Image to be distorted.

        Returns:
            Distorted image.
        """
        K = np.eye(3) * 1000
        K[0, 2] = img.shape[1] / 2
        K[1, 2] = img.shape[0] / 2
        K[2, 2] = 1

        return cv2.undistort(img, K, np.array([level, level, 0, 0]))

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = RadialDistortionPerturbation.distort(img, self.level)

        for key in results.get("seg_fields", []):
            results[key] = RadialDistortionPerturbation.distort(
                results[key], self.level
            )

        results["img"] = img
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level})"


@TRANSFORMS.register_module()
class CombinationPerturbation(BaseTransform):
    """Applies combination of other perturbations to the image.
    There are 5 configurations total."""

    def __init__(self, choice=0):
        self.choice = choice
        self.config = COMBINATION_PARAMETERS[choice]
        (
            self.b_alpha,
            self.g_alpha,
            self.r_alpha,
            self.h_alpha,
            self.s_alpha,
            self.v_alpha,
            self.blur_ksize,
            self.noise_sigma,
            _,
        ) = self.config

        self.blur_ksize = int(self.blur_ksize)

        self.r_direction = 0 if self.r_alpha < 0 else 1
        self.g_direction = 0 if self.g_alpha < 0 else 1
        self.b_direction = 0 if self.b_alpha < 0 else 1

        self.h_direction = 0 if self.h_alpha < 0 else 1
        self.s_direction = 0 if self.s_alpha < 0 else 1
        self.v_direction = 0 if self.v_alpha < 0 else 1

        self.rgb2hsv = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        self.hsv2rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_HSV2BGR)

    def transform(self, results: dict) -> dict:
        img = results["img"]

        # rgb perturbation
        img[..., 2] = (
            img[..., 2]
            - self.r_alpha * img[..., 2]
            + 255 * self.r_direction * self.r_alpha
        )
        img[..., 1] = (
            img[..., 1]
            - self.g_alpha * img[..., 1]
            + 255 * self.g_direction * self.g_alpha
        )
        img[..., 0] = (
            img[..., 0]
            - self.b_alpha * img[..., 0]
            + 255 * self.b_direction * self.b_alpha
        )

        # hsv perturbation
        img = self.rgb2hsv(img)
        img[..., 0] = (
            img[..., 0]
            - self.h_alpha * img[..., 0]
            + 180 * self.h_direction * self.h_alpha
        )
        img[..., 1] = (
            img[..., 1]
            - self.s_alpha * img[..., 1]
            + 255 * self.s_direction * self.s_alpha
        )

        if self.v_direction == 0:
            img[..., 2] = img[..., 2] - self.v_alpha * img[..., 2] + 10 * self.v_alpha
        else:
            img[..., 2] = img[..., 2] - self.v_alpha * img[..., 2] + 255 * self.v_alpha
        img = self.hsv2rgb(img)

        # blur perturbation
        img = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)

        # noise perturbation
        img = img + np.random.normal(0, self.noise_sigma, size=img.shape)

        results["img"] = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(choice={self.choice})"


def perturb_rgb(results, channel, alpha, direction):
    img = results["img"]
    img[..., channel] = (
        img[..., channel] - alpha * img[..., channel] + 255 * direction * alpha
    )
    results["img"] = img.astype(np.uint8)
    results["ori_shape"] = img.shape[:2]

    return results


def perturb_hsv(results, channel, alpha, direction):
    """HSV counterpart of `perturb_rgb`. channel 0/1/2 -> H/S/V, direction 0/1 ->
    darker/lighter.

    Lifted verbatim out of HSVPerturbation.transform so the two channel-specific
    special cases live in exactly one place: hue saturates at 180 under cv2's
    8-bit HSV (not 255), and darkening V rails toward a small positive floor
    rather than to 0, which would collapse the image to black and make the op
    indistinguishable from any other V darkening at high magnitude.

    Unlike `perturb_rgb` this does NOT rewrite results["ori_shape"] -- preserving
    HSVPerturbation's behaviour, which the legacy non-"_new" SA path in
    sensitivity_analysis.py/gpr_sa.py/bopt_sa.py constructs by name.
    """
    max_val = 180 if channel == 0 else 255

    img = cv2.cvtColor(results["img"], cv2.COLOR_BGR2HSV)
    if channel == 2 and direction == 0:
        img[..., channel] = img[..., channel] - alpha * img[..., channel] + 10 * alpha
    else:
        img[..., channel] = (
            img[..., channel] - alpha * img[..., channel] + max_val * direction * alpha
        )
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    results["img"] = img.astype(np.uint8)

    return results


@TRANSFORMS.register_module()
class LighterR(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=2, alpha=self.magnitude, direction=1)
        return results


@TRANSFORMS.register_module()
class DarkerR(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=2, alpha=self.magnitude, direction=0)
        return results


@TRANSFORMS.register_module()
class LighterG(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=1, alpha=self.magnitude, direction=1)
        return results


@TRANSFORMS.register_module()
class DarkerG(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=1, alpha=self.magnitude, direction=0)
        return results


@TRANSFORMS.register_module()
class LighterB(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=0, alpha=self.magnitude, direction=1)
        return results


@TRANSFORMS.register_module()
class DarkerB(BaseTransform):
    """Applies RGB perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        results = perturb_rgb(results, channel=0, alpha=self.magnitude, direction=0)
        return results


# The HSV half of the same vocabulary. HSVPerturbation already implemented these
# six, but keyed by (channel, alpha, direction) -- a signature the pdf-driven
# samplers cannot call, since RandomTrainTransformNew and _make_diff_transform
# both construct a transform as `cls(magnitude=level)`. These are the thin
# magnitude-signature wrappers that make lighter_H..darker_V addressable the same
# way lighter_R..darker_B already are.


@TRANSFORMS.register_module()
class LighterH(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=0, alpha=self.magnitude, direction=1)


@TRANSFORMS.register_module()
class DarkerH(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=0, alpha=self.magnitude, direction=0)


@TRANSFORMS.register_module()
class LighterS(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=1, alpha=self.magnitude, direction=1)


@TRANSFORMS.register_module()
class DarkerS(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=1, alpha=self.magnitude, direction=0)


@TRANSFORMS.register_module()
class LighterV(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=2, alpha=self.magnitude, direction=1)


@TRANSFORMS.register_module()
class DarkerV(BaseTransform):
    """Applies HSV perturbation to an image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)

    def transform(self, results: dict) -> dict:
        return perturb_hsv(results, channel=2, alpha=self.magnitude, direction=0)


@TRANSFORMS.register_module()
class Blur(BaseTransform):
    """Applies Gaussian blur perturbation assuming sigma in both X and Y directions are equal."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)
        self.kernel_size = int(magnitude * 49) // 2 * 2 + 1  # ensures odd

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

        results["img"] = img
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"


@TRANSFORMS.register_module()
class Noise(BaseTransform):
    """Applies Gaussian Noise perturbation to the image."""

    def __init__(self, magnitude: float):
        self.magnitude = abs(magnitude)
        self.mean = 0
        self.sigma = 50.0 * self.magnitude

    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = img + np.random.normal(self.mean, self.sigma, size=img.shape)

        results["img"] = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


LEGACY20_OPS_PHOTOMETRIC = {
    "BrightnessTransform": (BrightnessTransform, True),
    "NegativeBrightnessTransform": (NegativeBrightnessTransform, True),
    "ColorTransform": (ColorTransform, True),
    "NegativeColorTransform": (NegativeColorTransform, True),
    "ContrastTransform": (ContrastTransform, True),
    "NegativeContrastTransform": (NegativeContrastTransform, True),
    "SharpnessTransform": (SharpnessTransform, True),
    "NegativeSharpnessTransform": (NegativeSharpnessTransform, True),
    "PosterizeTransform": (PosterizeTransform, True),
    "SolarizeTransform": (SolarizeTransform, True),
}

LEGACY20_OPS_GEOMETRIC = {
    "ShearX": (ShearX, True),
    "ShearY": (ShearY, True),
    "NegativeShearX": (NegativeShearX, True),
    "NegativeShearY": (NegativeShearY, True),
    "TranslateX": (TranslateX, True),
    "TranslateY": (TranslateY, True),
    "NegativeTranslateX": (NegativeTranslateX, True),
    "NegativeTranslateY": (NegativeTranslateY, True),
    "Rotate": (Rotate, True),
    "NegativeRotate": (NegativeRotate, True),
}

LEGACY20_OPS = LEGACY20_OPS_PHOTOMETRIC | LEGACY20_OPS_GEOMETRIC


# The 32 GPU ops, split the same way the CPU registries are. Values here are the
# raw callables `(images, magnitude) -> images`, NOT the
# `(transform_cls, bool)` pairs the two CPU registries carry -- these ops are
# applied batched on GPU by sensaug.dataset.gpu_augment and never exist as
# pipeline transforms. resolve_perturbation_set documents the consequence.
DIFF32_OPS_GEOMETRIC = {
    name: op for name, op in DIFF32_OPS.items() if name in GEOMETRIC_OP_KEYS
}
DIFF32_OPS_PHOTOMETRIC = {
    name: op for name, op in DIFF32_OPS.items() if name not in GEOMETRIC_OP_KEYS
}


# --- the non-differentiable 32-op vocabulary ----------------------------------
#
# The same 32 op NAMES the gradient probe differentiates, played through the plain
# CPU transform classes above. The CPU counterpart of DIFF32_OPS.
#
# No --aug-type routes here any more: `ours`, `default` and `grad_corr` all train
# and run SA on "diff32", so that every compared arm applies exactly the function
# the probe measures. This set is kept because it is the reference the CPU/GPU
# equivalence test compares against, and because it remains selectable for a
# deliberate CPU run.
#
# The 1:1 name correspondence is not a coincidence to be re-derived: the
# differentiable registries were written to mirror these classes (see
# differentiable_augmentations.py's registry comment and
# differentiable_augmentations_aa.py's registry comment). The parity assertion
# below is what keeps them from drifting.
#
# NOT the same magnitude scale as the differentiable op of the same name. The two
# registries agree on WHICH op each name denotes, never on what magnitude 0.5 of
# it means. 24 of the 32 are calibrated; 8 are not -- `blur` (cv2's
# kernel-size-derived implicit sigma 7.70 here vs sigma 1.00 there, at m=1.0),
# `noise` (0.196 vs 1.00), and brightness/contrast/color +-, which differ in FORM
# rather than scale (see the TODO(calibration) table in
# differentiable_augmentations_aa.py). A per-op score read off R transfers between
# the two; a magnitude does not.
#
# Posterize/Solarize are absent: they are in LEGACY20_OPS but have no
# differentiable counterpart, so R cannot see them, so they do not belong in the
# vocabulary R is meant to govern.
NON_DIFF32_OPS = {
    # base 14 -- Shu et al.'s RGB/HSV/blur/noise ops
    "lighter_R": (LighterR, True),
    "darker_R": (DarkerR, True),
    "lighter_G": (LighterG, True),
    "darker_G": (DarkerG, True),
    "lighter_B": (LighterB, True),
    "darker_B": (DarkerB, True),
    "lighter_H": (LighterH, True),
    "darker_H": (DarkerH, True),
    "lighter_S": (LighterS, True),
    "darker_S": (DarkerS, True),
    "lighter_V": (LighterV, True),
    "darker_V": (DarkerV, True),
    "blur": (Blur, True),
    "noise": (Noise, True),
    # AutoAugment family -- these 18 ARE the LEGACY20_OPS classes, reached
    # under their differentiable-registry names.
    "rotate_pos": (Rotate, True),
    "rotate_neg": (NegativeRotate, True),
    "shear_x_pos": (ShearX, True),
    "shear_x_neg": (NegativeShearX, True),
    "shear_y_pos": (ShearY, True),
    "shear_y_neg": (NegativeShearY, True),
    "translate_x_pos": (TranslateX, True),
    "translate_x_neg": (NegativeTranslateX, True),
    "translate_y_pos": (TranslateY, True),
    "translate_y_neg": (NegativeTranslateY, True),
    "brightness_pos": (BrightnessTransform, True),
    "brightness_neg": (NegativeBrightnessTransform, True),
    "contrast_pos": (ContrastTransform, True),
    "contrast_neg": (NegativeContrastTransform, True),
    "sharpness_pos": (SharpnessTransform, True),
    "sharpness_neg": (NegativeSharpnessTransform, True),
    "color_pos": (ColorTransform, True),
    "color_neg": (NegativeColorTransform, True),
}

# Index-compatibility with R's axes is the entire premise; an op added to one
# registry and not the other would silently drop out of every per-op score
# (missing name -> no redundancy signal) rather than fail. Fail at import instead.
assert set(NON_DIFF32_OPS) == set(DIFF32_OPS), (
    "NON_DIFF32_OPS and DIFF32_OPS have diverged: "
    f"only in non-diff32={sorted(set(NON_DIFF32_OPS) - set(DIFF32_OPS))}, "
    f"only in diff32={sorted(set(DIFF32_OPS) - set(NON_DIFF32_OPS))}"
)

NON_DIFF32_OPS_GEOMETRIC = {
    name: value
    for name, value in NON_DIFF32_OPS.items()
    if name in GEOMETRIC_OP_KEYS
}
NON_DIFF32_OPS_PHOTOMETRIC = {
    name: value
    for name, value in NON_DIFF32_OPS.items()
    if name not in GEOMETRIC_OP_KEYS
}

def resolve_perturbation_set(
    name: str, geometric_only: bool = False, photometric_only: bool = False
) -> dict:
    """Select a perturbation registry by name, honouring the geometric/photometric
    filters.

    Three sets, differing on which op names are in play and how they are applied:

    - ``"legacy20"``  -- 20 PascalCase names, CPU transform classes. The historical
      set; keys ARE the registered mmseg type names.
    - ``"non-diff32"`` -- the 32 R-indexed snake_case names, CPU transform classes.
    - ``"diff32"``    -- the same 32 names as GPU-batched torch ops, applied by
      sensaug.dataset.gpu_augment rather than by any pipeline transform.

    VALUE SHAPES DIFFER. The two CPU sets map a name to ``(transform_cls, bool)``;
    ``diff32`` maps a name to the raw callable ``(images, magnitude) -> images``.
    Callers that only need the KEYS (the SA sweep, the pdf samplers) work against
    all three. Callers that need a transform class are CPU-only by construction
    and must never be handed ``diff32`` -- `_perturbation_transform_cfg` raises
    rather than unpacking a callable into a class.
    """
    if name == "legacy20":
        if geometric_only:
            return LEGACY20_OPS_GEOMETRIC
        if photometric_only:
            return LEGACY20_OPS_PHOTOMETRIC
        return LEGACY20_OPS

    if name == "non-diff32":
        if geometric_only:
            return NON_DIFF32_OPS_GEOMETRIC
        if photometric_only:
            return NON_DIFF32_OPS_PHOTOMETRIC
        return NON_DIFF32_OPS

    if name == GPU_PERTURBATION_SET:
        if geometric_only:
            return DIFF32_OPS_GEOMETRIC
        if photometric_only:
            return DIFF32_OPS_PHOTOMETRIC
        return DIFF32_OPS

    raise ValueError(
        f"unknown perturbation_set {name!r}, expected one of {PERTURBATION_SETS}"
    )

# @TRANSFORMS.register_module()
# class PackSegInputs(BaseTransform):
#     """Pack the inputs data for the semantic segmentation.

#     The ``img_meta`` item is always populated.  The contents of the
#     ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

#         - ``img_path``: filename of the image

#         - ``ori_shape``: original shape of the image as a tuple (h, w, c)

#         - ``img_shape``: shape of the image input to the network as a tuple \
#             (h, w, c).  Note that images may be zero padded on the \
#             bottom/right if the batch tensor is larger than this shape.

#         - ``pad_shape``: shape of padded images

#         - ``scale_factor``: a float indicating the preprocessing scale

#         - ``flip``: a boolean indicating if image flip transform was used

#         - ``flip_direction``: the flipping direction

#     Args:
#         meta_keys (Sequence[str], optional): Meta keys to be packed from
#             ``SegDataSample`` and collected in ``data[img_metas]``.
#             Default: ``('img_path', 'ori_shape',
#             'img_shape', 'pad_shape', 'scale_factor', 'flip',
#             'flip_direction')``
#     """

#     def __init__(self,
#                  meta_keys=('img_path', 'seg_map_path', 'ori_shape',
#                             'img_shape', 'pad_shape', 'scale_factor', 'flip',
#                             'flip_direction', 'reduce_zero_label')):
#         self.meta_keys = meta_keys

#     def transform(self, results: dict) -> dict:
#         """Method to pack the input data.

#         Args:
#             results (dict): Result dict from the data pipeline.

#         Returns:
#             dict:

#             - 'inputs' (obj:`torch.Tensor`): The forward data of models.
#             - 'data_sample' (obj:`SegDataSample`): The annotation info of the
#                 sample.
#         """
#         packed_results = dict()
#         if 'img' in results:
#             img = results['img']
#             if len(img.shape) < 3:
#                 img = np.expand_dims(img, -1)
#             if not img.flags.c_contiguous:
#                 img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
#             else:
#                 img = img.transpose(2, 0, 1)
#                 img = to_tensor(img).contiguous()
#             packed_results['inputs'] = img

#         data_sample = SegDataSample()
#         if 'gt_seg_map' in results:
#             if len(results['gt_seg_map'].shape) == 2:
#                 data = to_tensor(results['gt_seg_map'][None,
#                                                        ...].astype(np.int64))
#             else:
#                 warnings.warn('Please pay attention your ground truth '
#                               'segmentation map, usually the segmentation '
#                               'map is 2D, but got '
#                               f'{results["gt_seg_map"].shape}')
#                 data = to_tensor(results['gt_seg_map'].astype(np.int64))
#             gt_sem_seg_data = dict(data=data)
#             data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

#         if 'gt_edge_map' in results:
#             gt_edge_data = dict(
#                 data=to_tensor(results['gt_edge_map'][None,
#                                                       ...].astype(np.int64)))
#             data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

#         if 'gt_depth_map' in results:
#             gt_depth_data = dict(
#                 data=to_tensor(results['gt_depth_map'][None, ...]))
#             data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

#         img_meta = {}
#         for key in self.meta_keys:
#             if key in results:
#                 img_meta[key] = results[key]
#         data_sample.set_metainfo(img_meta)
#         packed_results['data_samples'] = data_sample

#         return packed_results

#     def __repr__(self) -> str:
#         repr_str = self.__class__.__name__
#         repr_str += f'(meta_keys={self.meta_keys})'
#         return repr_str
