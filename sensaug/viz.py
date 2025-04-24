import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage


def _unnormalize_imagenet_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.mul_(std).add_(mean)  # normalize for network

    return img


def plot_fig1_eccv(root, img_basenames):
    img_stack = []

    for b in img_basenames:
        img_orig = read_image(os.path.join(root, b + "_orig.png"))
        img_aug = read_image(os.path.join(root, b + "_aug.png"))

        # img_orig = (_unnormalize_imagenet_img(img_orig / 255) * 255).byte()
        # img_aug = (_unnormalize_imagenet_img(img_aug / 255) * 255).byte()

        img_stack.append(img_orig)
        img_stack.append(img_aug)

    img_stack = img_stack[0::2] + img_stack[1::2]  # orig on top, aug on bottom
    grid = make_grid(img_stack, nrow=(len(img_stack) // 2))
    # print(grid)
    img = ToPILImage()(grid)
    img.save("figs/optimization_grid.png")


if __name__ == "__main__":
    img_root = "/scratch/vision_robustness/figs/temp"

    fig_names = ["iter_99", "iter_499", "iter_699", "iter_899", "iter_1999"]

    plot_fig1_eccv(img_root, fig_names)
