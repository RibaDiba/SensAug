import cv2
import numpy as np
from PIL import Image


def bound_image(image, aspect_ratio):
    """
    Shrinks a scaled rectangle within the image bounds until all corners reach a non-zero pixel.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape
    x_min, x_max, y_min, y_max = 0, w, 0, h

    if x_min >= x_max - 1 or y_min >= y_max - 1:
        return (0, 0, h, w), False

    corners = [
        (y_min, x_min),
        (y_min, x_max - 1),
        (y_max - 1, x_max - 1),
        (y_max - 1, x_min),
    ]
    corner_pixels = [gray_image[c[0], c[1]] for c in corners]

    while any(pix <= 0 for pix in corner_pixels):
        new_h = (y_max - y_min) - 2
        new_w = int(new_h * aspect_ratio)
        delta_w = (x_max - x_min) - new_w

        if corner_pixels[0] == 0 or corner_pixels[3] == 0:
            x_min += delta_w // 2
        if corner_pixels[1] == 0 or corner_pixels[2] == 0:
            x_max -= delta_w // 2

        y_min += 1
        y_max -= 1

        if x_min >= x_max - 10 or y_min >= y_max - 10:
            return (0, 0, h, w), False

        corners = [
            (y_min, x_min),
            (y_min, x_max - 1),
            (y_max - 1, x_max - 1),
            (y_max - 1, x_min),
        ]
        corner_pixels = [gray_image[c[0], c[1]] for c in corners]

    return (y_min, x_min, y_max - y_min, x_max - x_min), (y_max - y_min) > 0 and (
        x_max - x_min
    ) > 0


def crop_and_resize_data(input_data, original_h=None, original_w=None):
    if isinstance(input_data, Image.Image):
        input_data = np.array(input_data)
    else:
        input_data = (
            input_data.copy()
        )  # don't modify the original data in case bounds aren't valid

    if original_h is None or original_w is None:
        original_h, original_w = input_data.shape[:2]

    aspect_ratio = original_w / original_h
    cropped_rect, valid_bounds = bound_image(input_data, aspect_ratio)

    if valid_bounds:
        y, x, h, w = cropped_rect
        cropped_data = input_data[y : y + h, x : x + w]
        interpolation = (
            cv2.INTER_NEAREST if len(input_data.shape) == 2 else cv2.INTER_LINEAR
        )
        resized_data = cv2.resize(
            cropped_data, (original_w, original_h), interpolation=interpolation
        )
        return resized_data, cropped_rect
    else:
        return input_data, None


def apply_cropping_to_labels(results: dict, cropped_rect: tuple):
    if results.get("gt_seg_map", None) is not None and cropped_rect is not None:
        y, x, h, w = cropped_rect

        if h <= 0 or w <= 0:
            return results

        label = results["gt_seg_map"]
        orig_h, orig_w = label.shape[:2]
        cropped_label = cv2.resize(
            label,
            (results["img"].shape[1], results["img"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        cropped_label = cropped_label[y : y + h, x : x + w]
        cropped_label = cv2.resize(
            cropped_label, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        results["gt_seg_map"] = cropped_label
        h, w = results["img"].shape[:2]
        # assert cropped_label.shape[0] == h and cropped_label.shape[1] == w, f"segmentation map size {cropped_label.shape} does not match image shape {(h, w)}"

    return results
