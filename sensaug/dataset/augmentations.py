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
from sensaug.dataset.differentiable_augmentations import (
    DIFFERENTIABLE_PERTURBATIONS,
    img_to_rgb01,
    rgb01_to_img,
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
        perturbation_set: str = "new",
    ):
        self.geometric_only = geometric_only
        self.photometric_only = photometric_only
        # "new" -> NEW_PERTURBATIONS, "diff" -> the differentiable ops the gradient
        # cross-correlation pipeline measures. A string rather than the dict itself
        # because this goes through an mmengine Config, which cannot serialize
        # classes. Validated eagerly so a bad value fails at build time, not on the
        # first sampled image inside a dataloader worker.
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
    Randomly selects a perturbation from NEW PERTURBATIONS list.
    Then, uses a beta distribution to mix the perturbed image with the original image.
    """

    def __init__(self, pdf_dict: dict, perturbation_set: str = "new"):
        self.pdf_dict: dict = pdf_dict
        # Which registry the pdf's op names are drawn from -- "new" for
        # NEW_PERTURBATIONS, "diff" for the differentiable ops. The two share no
        # names, so a mismatch is a KeyError on the first augmented image rather
        # than anything subtle. See resolve_perturbation_set.
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

        max_val = 180 if self.channel == 0 else 255

        img = results["img"]
        img = self.rgb2hsv(img)
        if self.channel == 2 and self.direction == 0:
            img[..., self.channel] = (
                img[..., self.channel]
                - self.alpha * img[..., self.channel]
                + 10 * self.alpha
            )
        else:
            img[..., self.channel] = (
                img[..., self.channel]
                - self.alpha * img[..., self.channel]
                + max_val * self.direction * self.alpha
            )
        img = self.hsv2rgb(img)

        results["img"] = img.astype(np.uint8)

        return results

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


NEW_PERTURBATIONS_PHOTOMETRIC = {
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

NEW_PERTURBATIONS_GEOMETRIC = {
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

NEW_PERTURBATIONS = NEW_PERTURBATIONS_PHOTOMETRIC | NEW_PERTURBATIONS_GEOMETRIC


# --- differentiable-op pipeline wrappers -------------------------------------
#
# The sensitivity analysis can only measure an augmentation that exists as a
# registered pipeline transform: it works by inserting one into the val
# dataloader and re-evaluating (loops.RobustValLoop.test_perturbed_new ->
# runner_utils.apply_perturbations_dataloader). The ops in
# sensaug.dataset.differentiable_augmentations are plain torch functions, so SA
# structurally could not see them -- which is why the SA curve was keyed by an
# entirely disjoint vocabulary (NEW_PERTURBATIONS above) from the one the
# gradient probe differentiates, and `sa_curve["lighter_R"]` never existed.
#
# These wrappers close that gap. SA and the probe now score the SAME function,
# not two implementations that share a name, and the SA levels land natively in
# the probe's magnitude units ([0, 1]) with no rescaling -- unlike the legacy
# PERTURBATIONS vocabulary, whose blur level is a kernel size in [0, 49] and
# whose noise level is a sigma in [0, 50].


class _DiffAugTransform(BaseTransform):
    """Base for the generated per-op wrappers. Subclasses set ``OP_NAME``.

    Takes ``magnitude`` (not ``delta``/``sigma``) so the constructor signature
    matches NEW_PERTURBATIONS' classes -- that is what lets RandomTrainTransformNew
    and the SA loop swap registries with a lookup change instead of a special case.

    These ops do not change image shape, so ``ori_shape`` is deliberately left
    alone (unlike Blur/Noise above, which rewrite it redundantly).
    """

    OP_NAME: str = ""

    def __init__(self, magnitude: float):
        self.magnitude = float(magnitude)
        self.op = DIFFERENTIABLE_PERTURBATIONS[self.OP_NAME]

    def transform(self, results: dict) -> dict:
        # no_grad: inside a dataloader worker there is nothing to differentiate,
        # and the graph would otherwise be built and discarded per sample.
        with torch.no_grad():
            perturbed = self.op(img_to_rgb01(results["img"]), self.magnitude)
        results["img"] = rgb01_to_img(perturbed)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(magnitude={self.magnitude})"


def _make_diff_transform(name: str):
    """Generate + register one wrapper class for `name`.

    Generated rather than hand-written because 14 near-identical classes would
    drift. The class is bound into module globals under its generated name so
    pickle can find it: dataloader workers started with `spawn` (rather than
    Linux's default `fork`) pickle the dataset, and a class unreachable by
    qualified name would fail there and nowhere else.
    """
    cls_name = "Diff" + "".join(part.capitalize() for part in name.split("_"))
    cls = type(
        cls_name,
        (_DiffAugTransform,),
        {
            "OP_NAME": name,
            "__doc__": f"MMSeg pipeline wrapper for the differentiable `{name}` op.",
        },
    )
    globals()[cls_name] = cls
    return TRANSFORMS.register_module()(cls)


# Mirrors NEW_PERTURBATIONS' {name: (transform_cls, bool)} shape. The trailing
# flag is unused there too (every read site does `transform_cls, _ = ...`); it is
# carried so the two registries are interchangeable at every lookup site.
DIFF_PERTURBATIONS = {
    name: (_make_diff_transform(name), True) for name in DIFFERENTIABLE_PERTURBATIONS
}


def resolve_perturbation_set(
    name: str, geometric_only: bool = False, photometric_only: bool = False
) -> dict:
    """Select a perturbation registry by name, honouring the geometric/photometric
    filters. Both registries share the {op_name: (transform_cls, bool)} shape, so
    callers need only swap the lookup.

    Called at runtime (not at import) because DIFF_PERTURBATIONS is defined below
    the transforms that use it.
    """
    if name == "new":
        if geometric_only:
            return NEW_PERTURBATIONS_GEOMETRIC
        if photometric_only:
            return NEW_PERTURBATIONS_PHOTOMETRIC
        return NEW_PERTURBATIONS

    if name == "diff":
        if geometric_only:
            # Every differentiable op is photometric -- there is no differentiable
            # shear/translate/rotate. Falling through to "all of them" would hand
            # back a photometric-only set under a geometric-only flag, which is a
            # silently wrong experiment rather than an error.
            raise ValueError(
                "geometric_only is not available for the 'diff' perturbation set: "
                f"none of {sorted(DIFF_PERTURBATIONS)} are geometric"
            )
        return DIFF_PERTURBATIONS

    raise ValueError(
        f"unknown perturbation_set {name!r}, expected 'new' or 'diff'"
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
