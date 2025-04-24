# adapted from https://github.com/DeepVoltaire/AutoAugment/blob/master/ops.py

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps


class ShearX:
    def __call__(self, results, magnitude):
        shear_factor = magnitude * np.random.choice([-1, 1])
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)

        for key in ["img"] + results.get("seg_fields", []):
            img_array = results[key]
            h, w = img_array.shape[:2]
            transformed_img = cv2.warpAffine(
                img_array, M, (w, h), borderValue=(128, 128, 128)
            )
            results[key] = transformed_img

        return results


class ShearY:
    def __call__(self, results, magnitude):
        shear_factor = magnitude * np.random.choice([-1, 1])
        M = np.array([[1, 0, 0], [shear_factor, 1, 0]], dtype=np.float32)

        for key in ["img"] + results.get("seg_fields", []):
            img_array = results[key]
            h, w = img_array.shape[:2]
            transformed_img = cv2.warpAffine(
                img_array, M, (w, h), borderValue=(128, 128, 128)
            )
            results[key] = transformed_img

        return results


class TranslateX:
    def __call__(self, results, magnitude):
        tx = magnitude * np.random.choice([-1, 1])
        M = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)

        for key in ["img"] + results.get("seg_fields", []):
            img_array = results[key]
            h, w = img_array.shape[:2]
            transformed_img = cv2.warpAffine(
                img_array, M, (w, h), borderValue=(128, 128, 128)
            )
            results[key] = transformed_img

        return results


class TranslateY:
    def __call__(self, results, magnitude):
        ty = magnitude * np.random.choice([-1, 1])
        M = np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32)

        for key in ["img"] + results.get("seg_fields", []):
            img_array = results[key]
            h, w = img_array.shape[:2]
            transformed_img = cv2.warpAffine(
                img_array, M, (w, h), borderValue=(128, 128, 128)
            )
            results[key] = transformed_img

        return results


class Rotate:
    def __call__(self, results, magnitude):
        angle = magnitude * np.random.choice([-1, 1])
        for key in ["img"] + results.get("seg_fields", []):
            img_array = results[key]
            h, w = img_array.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_img = cv2.warpAffine(
                img_array, M, (w, h), borderValue=(128, 128, 128)
            )
            results[key] = rotated_img

        return results


class Color:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        enhancer = ImageEnhance.Color(img)
        img_enhanced = enhancer.enhance(1 + magnitude * np.random.choice([-1, 1]))
        results["img"] = np.array(img_enhanced)
        return results


class Posterize:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        pil_img = ImageOps.posterize(img, int(magnitude))
        results["img"] = np.array(pil_img)
        return results


class Solarize:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        pil_img = ImageOps.solarize(img, int(magnitude))
        results["img"] = np.array(pil_img)
        return results


class Contrast:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(1 + magnitude * np.random.choice([-1, 1]))
        results["img"] = np.array(img_enhanced)
        return results


class Brightness:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(1 + magnitude * np.random.choice([-1, 1]))
        results["img"] = np.array(img_enhanced)
        return results


class Sharpness:
    def __call__(self, results, magnitude):
        img = Image.fromarray(results["img"])
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(1 + magnitude * np.random.choice([-1, 1]))
        results["img"] = np.array(img_enhanced)
        return results


class Invert:
    def __call__(self, results, magnitude=None):
        img = Image.fromarray(results["img"])
        img_inverted = ImageOps.invert(img)
        results["img"] = np.array(img_inverted)
        return results


class Equalize:
    def __call__(self, results, magnitude=None):
        img = Image.fromarray(results["img"])
        img_equalized = ImageOps.equalize(img)
        results["img"] = np.array(img_equalized)
        return results


class AutoContrast:
    def __call__(self, results, magnitude=None):
        img = Image.fromarray(results["img"])
        img_autocontrasted = ImageOps.autocontrast(img)
        results["img"] = np.array(img_autocontrasted)
        return results
