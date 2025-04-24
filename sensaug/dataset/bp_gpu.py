# functions to perturb a batch of images

import torch
import kornia


def diff_rgb_aug(images, channel, alpha):
    """
    Change one of the rgb channels of a batch of images to be a weighted average
    of the the true value and either 0 or 255

    Parameters
    ----------
    images : Tensor
        The batch of images to be perturbed in the shape (B, H, W, 3). Channels
        are in the order RGB.
    channel : int
        The channel to perturb. 0 is R, 1 is G, 2 is B.
    direction : int
        0 to darken, 1 to brighten.
    alpha : float
        Severity parameter for perturbation.

    Returns
    -------
    perturbed_images : Tensor
        Perturbed images of shape (B, H, W, 3)
    """

    assert channel == 0 or channel == 1 or channel == 2, (
        "channel must be out of [0, 1, 2]"
    )

    alpha = alpha.clamp(min=-1.0, max=1.0)
    direction = (alpha > 0).int()
    alpha = torch.abs(alpha)

    perturbed_images = images.clone()

    if perturbed_images.dim() == 4:  # (B, 3, H, W)
        perturbed_images[:, channel] = (
            images[:, channel] - alpha * images[:, channel] + 1 * direction * alpha
        )
    else:  # (3, H, W)
        perturbed_images[channel] = (
            images[channel] - alpha * images[channel] + 1 * direction * alpha
        )

    return perturbed_images


def hsv_gpu_perturb(images, channel, direction, alpha):
    """
    Change one of the hsv channels of a batch of images (given in rgb) to be a
    weighted average of the the true value and either 0 or 255 (179 instead of
    255 for hue). In the special case of lowering the value channel, the
    weighted average is computed using the true values and 10 instead of 0.

    Parameters
    ----------
    images : Tensor
        The batch of images to be perturbed in the shape (B, H, W, 3). Channels
        are in the order RGB.
    channel : int
        The channel to perturb. 0 is hue, 1 is saturation, 2 is value.
    direction : int
        0 to darken, 1 to brighten.
    alpha : float
        Severity parameter for perturbation.

    Returns
    -------
    perturbed_images : Tensor
        Perturbed images of shape (B, H, W, 3), given in rgb
    """

    assert channel == 0 or channel == 1 or channel == 2, (
        "channel must be out of [0, 1, 2]"
    )
    assert alpha >= 0 and alpha <= 1, "alpha must be any value from 0 to 1"
    assert direction == 0 or direction == 1, (
        "direction must be either 0 (darker) or 1 (lighter)"
    )

    hsv_images = rgb2hsv_torch(images.permute(0, 3, 1, 2) / 255)

    # if lowering value, compute weighted average with 10 instead of 0
    if channel == 2 and direction == 0:
        hsv_images[:, channel, :, :] = (
            hsv_images[:, channel, :, :]
            - alpha * hsv_images[:, channel, :, :]
            + (10 / 255) * alpha
        )
    else:
        hsv_images[:, channel, :, :] = (
            hsv_images[:, channel, :, :]
            - alpha * hsv_images[:, channel, :, :]
            + direction * alpha
        )

    return (hsv2rgb_torch(hsv_images).permute(0, 2, 3, 1) * 255).to(torch.uint8)


# reference: https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html
from torchvision.transforms import GaussianBlur


def diff_blur_aug(images, kernel_size=(67, 67), sigma=1.0):
    """
    Apply a Gaussian blur to each image in images.

    Parameters
    ----------
    images : Tensor
        The batch of images to be perturbed in the shape (B, H, W, 3). Channels
        are in the order RGB.
    kernel_size : list
        Size of the kernel. Dimensions must be odd.
    sigma : list
        Min and max values of the standard deviation used to create the kernel.

    Returns
    -------
    perturbed_images : Tensor
        Perturbed images of shape (B, H, W, 3), given in rgb
    """

    # blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    images = kornia.filters.gaussian_blur2d(images, kernel_size, sigma, separable=False)

    # blur takes in input with shape [...,H,W]
    # return blur(images.permute(0,3,1,2)).permute(0,2,3,1).to(torch.uint8)
    return images


def diff_noise_aug(images, sigma):
    """
    Add zero mean Gaussian noise to each pixel for each image in images.

    Parameters
    ----------
    images : Tensor
        The batch of images to be perturbed in the shape (B, H, W, 3). Channels
        are in the order RGB.
    sigma : float
        Standard deviation of the noise to add.

    Returns
    -------
    perturbed_images : Tensor
        Perturbed images of shape (B, H, W, 3), given in rgb
    """

    noise = torch.randn(images.shape).cuda() * sigma
    noise = noise.to(images.device)

    return torch.clamp((images + noise), min=0, max=1.0)


# these functions taken from https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
# important notes for these functions:
#   - inputs are assumed to be shape (batch size, channels, height, width)
#   - input channels are assumed to have values from 0 to 1 (instead of 0 to 255)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb
