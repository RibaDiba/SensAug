import math
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation

from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.transforms import RandomFlip
from mmcv.transforms.base import BaseTransform


def apply_torch_fn_to_data(img, f):
    img_tensor = mmseg2tensor(img)
    img_tensor = f(img_tensor)
    final_img = tensor2mmseg(img_tensor)
    return np.ascontiguousarray(final_img)


def mmseg2tensor(img: np.ndarray) -> torch.Tensor:
    img_th = torch.tensor(img.copy())
    if len(img.shape) == 3:
        img_th = img_th.permute(2, 0, 1)
        img_th = img_th.flip(dims=(0,))  # BGR -> RGB
    return img_th


def tensor2mmseg(img: torch.Tensor) -> np.ndarray:
    if len(img.shape) == 3:
        img = img.flip(dims=(0,))  # RGB -> BGR
        img = img.permute(1, 2, 0)

    img_np = img.numpy().astype(np.uint8)
    return np.ascontiguousarray(img_np)


@TRANSFORMS.register_module()
class IDBHTransform(BaseTransform):
    def __init__(self, version="cifar10-weak"):
        super().__init__()
        if version == "cifar10-weak":
            layers: list[BaseTransform] = [
                RandomFlip(0.5, direction="horizontal"),
                CropShift(0, 11),
                ColorShape("color"),
                RandomErasing(p=0.5),
            ]
        elif version == "cifar10-strong":
            layers = [
                RandomFlip(0.5, direction="horizontal"),
                CropShift(0, 11),
                ColorShape("color"),
                RandomErasing(p=1),
            ]
        elif version == "svhn":
            layers = [
                RandomFlip(0.5, direction="horizontal"),
                CropShift(0, 9),
                ColorShape("shape"),
                RandomErasing(p=1, scale=(0.02, 0.5)),
            ]
        else:
            raise Exception("IDBH: invalid version string")

        self.layers: list = layers

    def transform(self, results: dict) -> dict:
        for l in self.layers:
            results = l(results)

        return results


class ColorShape(BaseTransform):
    ColorBiased = [
        (0.125, "color", 0.1, 1.9),
        (0.125, "brightness", 0.5, 1.9),
        (0.125, "contrast", 0.5, 1.9),
        (0.125, "sharpness", 0.1, 1.9),
        (0.125, "autocontrast"),
        (0.125, "equalize"),
        (0.125, "shear", 0.05, 0.15),
        (0.125, "rotate", 1, 11),
    ]
    ShapeBiased = [
        (0.08, "color", 0.1, 1.9),
        (0.08, "brightness", 0.5, 1.9),
        (0.04, "contrast", 0.5, 1.9),
        (0.08, "sharpness", 0.1, 1.9),
        (0.04, "autocontrast"),
        (0.08, "equalize"),
        (0.3, "shear", 0.05, 0.35),
        (0.3, "rotate", 1, 31),
    ]

    def __init__(self, version="color"):
        super().__init__()

        assert version in ["color", "shape"]
        space = self.ColorBiased if version == "color" else self.ShapeBiased

        self.space = {}
        p_accu = 0.0
        for trans in space:
            p = trans[0]
            self.space[(p_accu, p_accu + p)] = trans[1:]
            p_accu += p

    def apply(self, results, trans):
        img = results["img"]
        label = results.get("gt_seg_map", None)
        assert label is not None, (
            "labels are not loaded yet, cannot apply augmentations"
        )

        label = np.expand_dims(
            label, -1
        )  # add a "fake" channel to label for use with torchvision

        strength = 0

        if len(trans) == 1:
            trans = trans[0]
        else:
            lower, upper = trans[1:]
            trans = trans[0]
            if trans == "rotate":
                strength = torch.randint(lower, upper, (1,)).item()
            else:
                strength = torch.rand(1) * (upper - lower) + lower

        if trans == "color":
            img = apply_torch_fn_to_data(
                img, lambda x: F.adjust_saturation(x, strength)
            )
        elif trans == "brightness":
            img = apply_torch_fn_to_data(
                img, lambda x: F.adjust_brightness(x, strength)
            )
        elif trans == "contrast":
            img = apply_torch_fn_to_data(img, lambda x: F.adjust_contrast(x, strength))
        elif trans == "sharpness":
            img = apply_torch_fn_to_data(img, lambda x: F.adjust_sharpness(x, strength))
        elif trans == "shear":
            if torch.randint(2, (1,)):
                # random sign
                strength *= -1
            strength = math.degrees(strength)
            strength = [strength, 0.0] if torch.randint(2, (1,)) else [0.0, strength]
            img = apply_torch_fn_to_data(
                img,
                lambda x: F.affine(
                    x,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=strength,
                    interpolation=Interpolation.NEAREST,
                    fill=0,
                ),
            )  # type: ignore
            label = apply_torch_fn_to_data(
                label,
                lambda x: F.affine(
                    x,
                    angle=0.0,
                    translate=[0, 0],
                    scale=1.0,
                    shear=strength,
                    interpolation=Interpolation.NEAREST,
                    fill=255,
                ),
            )  # type: ignore
        elif trans == "rotate":
            if torch.randint(2, (1,)):
                strength *= -1
            img = apply_torch_fn_to_data(
                img,
                lambda x: F.rotate(
                    x, angle=strength, interpolation=Interpolation.NEAREST, fill=0
                ),
            )  # type: ignore
            label = apply_torch_fn_to_data(
                label,
                lambda x: F.rotate(
                    x, angle=strength, interpolation=Interpolation.NEAREST, fill=255
                ),
            )  # type: ignore
        elif trans == "autocontrast":
            img = apply_torch_fn_to_data(img, lambda x: F.autocontrast(x))
        elif trans == "equalize":
            img = apply_torch_fn_to_data(img, lambda x: F.equalize(x))

        results["img"] = img
        results["gt_seg_map"] = label[:, :, 0]

        return results

    def transform(self, results):
        roll = torch.rand(1)
        for (lower, upper), trans in self.space.items():
            if roll <= upper and roll >= lower:
                return self.apply(results, trans)

        return results


class CropShift(BaseTransform):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)

    def sample_top(self, x_top: int, y_top: int) -> tuple[int, int]:
        x: int = torch.randint(0, x_top + 1, (1,)).item()  # type: ignore
        y: int = torch.randint(0, y_top + 1, (1,)).item()  # type: ignore
        return x, y

    def transform(self, results: dict) -> dict:
        img = results["img"]
        label = results.get("gt_seg_map", None)

        assert label is not None, (
            "label not in data; cannot perform augmentation cropshift"
        )
        orig_img_shape = img.shape
        orig_label_shape = label.shape

        img = mmseg2tensor(img)
        label = mmseg2tensor(label)

        strength: int = torch.randint(self.low, self.high, (1,)).item()  # type: ignore

        if self.low == self.high:
            strength = self.low

        w, h = F.get_image_size(img)
        crop_x: int = torch.randint(0, strength + 1, (1,)).item()  # type: ignore
        crop_y: int = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)

        img = F.crop(img, top_y, top_x, crop_h, crop_w)
        img = F.pad(img, padding=[crop_x, crop_y], fill=0)

        top_x, top_y = self.sample_top(crop_x, crop_y)

        cropshift_img_pil = F.crop(img, top_y, top_x, h, w)
        cropshift_img = tensor2mmseg(cropshift_img_pil)

        cropshift_label_pil = F.crop(label, top_y, top_x, h, w)
        cropshift_label = tensor2mmseg(cropshift_label_pil)

        assert cropshift_img.shape == orig_img_shape, (
            f"cropshift img shape mismatch: {orig_img_shape} (orig) vs. {cropshift_img.shape} (new)"
        )
        assert cropshift_label.shape == orig_label_shape, (
            f"cropshift img shape mismatch: {orig_label_shape} (orig) vs. {cropshift_label.shape} (new)"
        )

        results["img"] = cropshift_img
        results["gt_seg_map"] = cropshift_label
        # results["img_shape"] = results["img"].shape[:2]
        # results["pad_shape"] = results["img"].shape[:2]
        # results["ori_shape"] = results["img"].shape[:2]

        return results


class RandomErasing(BaseTransform):
    """Erases part of the image. Only modifies the image input, not the label map."""

    def __init__(self, p=0.5, scale=(0.02, 0.33)):
        self.p = p
        self.scale = scale
        self.layer = T.RandomErasing(p=p, scale=scale)

    def transform(self, results: dict):
        results["img"] = apply_torch_fn_to_data(results["img"], self.layer)
        return results
