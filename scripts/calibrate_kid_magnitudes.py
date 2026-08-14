#!/usr/bin/env python
"""Calibrate a per-op probe magnitude via KID (Kernel Inception Distance) so
every differentiable augmentation carries roughly the same visual-distortion
budget, for runs that have no sensitivity-analysis snapshot to draw a
commensurate magnitude from.

CollectGradientHook (sensaug/hooks/grad_hook.py) falls back to one hardcoded
scalar, ref_magnitude=0.5, for every op whenever there is no SA-published
distribution -- and that constant is not commensurate across ops (magnitude
0.5 means "mild blur" for `blur` but "saturates most pixels" for `noise`).
This script picks, per op, the magnitude in [0, 1] whose KID against a clean
reference image set is closest to a shared target KID -- so different ops are
held at the same distortion strength instead of the same raw number.

The output is a corr_magnitudes.json-shaped seed file: CollectGradientHook
already knows how to read this exact shape via --magnitudes-path
(_load_seed_snapshot in grad_hook.py), so nothing about the hooks,
corr_magnitudes.py, or the live training pipeline changes. This script is
self-contained: it only reads a finished experiment's dumped config (for the
val set's image directory) and writes its own output file.

Usage:
    python scripts/calibrate_kid_magnitudes.py --work-dir experiments/none_pspnet_cityscapes
    python scripts/compute_grad_corr.py --work-dir experiments/none_pspnet_cityscapes \\
        --magnitudes-path experiments/none_pspnet_cityscapes/corr_magnitudes_kid_seed.json
"""

import argparse
import glob
import json
import os
import sys

# Repo root, so `sensaug` resolves whether this is run as
# `python scripts/calibrate_kid_magnitudes.py` from the repo root or from
# elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from mmengine.config import Config
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.kid import KernelInceptionDistance

from sensaug.dataset.differentiable_augmentations_aa import (
    ALL_DIFFERENTIABLE_PERTURBATIONS,
)

IMG_EXTENSIONS = {"png", "jpg"}


class ImagePathDataset(Dataset):
    """Loads every image under a directory as an (H, W, 3) RGB uint8 array.

    Inlined rather than imported from sensaug/metrics/kid.py (which defines
    the same class): that module's own imports are currently broken --
    sensaug/dataset/bp_gpu.py no longer exports the names it expects
    (rgb_gpu_perturb/blur_gpu_perturb/noise_gpu_perturb vs. that file's
    current diff_rgb_aug/diff_blur_aug/diff_noise_aug) -- and this script is
    meant to be self-contained regardless.
    """

    def __init__(self, path):
        """
        Initialize the dataset with image files found recursively under a directory.
        
        Parameters:
            path (str or pathlib.Path): Root directory to search for image files.
        """
        self.images = [
            file for ext in IMG_EXTENSIONS for file in Path(path).rglob(f"*.{ext}")
        ]

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, i):
        """
        Load and return the image at the specified dataset index.
        
        Parameters:
        	i (int): Index of the image to load.
        
        Returns:
        	numpy.ndarray: The image as an RGB array.
        """
        img = cv2.imread(str(self.images[i]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_args():
    """
    Parse command-line options for KID-based augmentation magnitude calibration.
    
    Returns:
    	Namespace: Parsed calibration settings, including the experiment directory, reference data, target KID, candidate grid, operations, batch size, output path, and device.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a per-op probe magnitude via KID, for use as a "
            "commensurate-across-ops replacement of the fixed ref_magnitude "
            "fallback."
        )
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="finished experiment directory (holds the dumped config, for the val set path)",
    )
    parser.add_argument(
        "--kid-reference-dir",
        default=None,
        help="clean reference image directory; default is the val set from the dumped config",
    )
    parser.add_argument(
        "--target-kid-from",
        default="noise@0.5",
        help=(
            "shared target KID, derived by running this op at this magnitude "
            "once ('<op>@<magnitude>'); overridden by --target-kid if given"
        ),
    )
    parser.add_argument(
        "--target-kid",
        type=float,
        default=None,
        help="explicit shared target KID value; skips --target-kid-from",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="number of candidate magnitudes in [0.05, 1.0] evaluated per op",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        help="subset of op names to calibrate; default is all differentiable ops",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="images per KID subset (matches sensaug/metrics/kid.py::KID's default)",
    )
    parser.add_argument("--output", default=None, help="default is <work-dir>/corr_magnitudes_kid_seed.json")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_op_at_magnitude(spec):
    """
    Parse an operation and magnitude specification.
    
    Parameters:
    	spec (str): An operation name followed by `@` and its magnitude.
    
    Returns:
    	tuple: The operation name and magnitude as a two-item tuple.
    
    Raises:
    	ValueError: If the operation is not supported.
    """
    op, _, magnitude_str = spec.partition("@")
    if op not in ALL_DIFFERENTIABLE_PERTURBATIONS:
        raise ValueError(
            f"unknown op {op!r} in --target-kid-from; choose from "
            f"{sorted(ALL_DIFFERENTIABLE_PERTURBATIONS)}"
        )
    return op, float(magnitude_str)


def kid_at(kid_metric, dataloader, op_name, magnitude, device):
    """
    Measure the KID produced by applying an augmentation at a specified magnitude.
    
    Parameters:
    	op_name (str): Name of the differentiable augmentation to apply.
    	magnitude (float): Augmentation magnitude.
    	dataloader: Batches of reference images to augment and evaluate.
    
    Returns:
    	float: Mean KID between the accumulated reference features and the augmented images.
    """
    op = ALL_DIFFERENTIABLE_PERTURBATIONS[op_name]
    kid_metric.reset()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            rgb01 = batch.permute(0, 3, 1, 2).float() / 255.0
            # Ops require a Tensor delta (they reshape it internally to
            # broadcast against the batch) -- a bare Python float is not
            # accepted despite the Union[float, Tensor] type hint.
            delta = torch.tensor(float(magnitude), dtype=rgb01.dtype, device=rgb01.device)
            perturbed01 = op(rgb01, delta)
            perturbed_uint8 = (perturbed01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            kid_metric.update(perturbed_uint8, real=False)
    mean, _std = kid_metric.compute()
    return float(mean)


def calibrate_op(kid_metric, dataloader, op_name, target_kid, grid, device):
    """
    Selects the operation magnitude whose KID is closest to the target.
    
    Parameters:
    	target_kid (float): KID value to match.
    	grid (array-like): Candidate magnitudes to evaluate.
    
    Returns:
    	tuple[float, float]: The selected magnitude and its KID score.
    """
    scores = np.array(
        [kid_at(kid_metric, dataloader, op_name, m, device) for m in grid]
    )
    best = int(np.argmin(np.abs(scores - target_kid)))
    return float(grid[best]), float(scores[best])


def main():
    """
    Calibrate augmentation magnitudes against a shared target KID and write the resulting seed configuration.
    
    Raises:
    	FileNotFoundError: If no dumped configuration or reference images are found, or if the dataset has fewer images than the batch size.
    	ValueError: If the batch size produces no complete batches or an unknown operation is requested.
    """
    args = parse_args()
    work_dir = os.path.abspath(args.work_dir)

    config_paths = glob.glob(os.path.join(work_dir, "*.py"))
    if not config_paths:
        raise FileNotFoundError(f"no dumped config (*.py) found in {work_dir}")
    cfg = Config.fromfile(config_paths[0])

    if args.kid_reference_dir is not None:
        reference_dir = args.kid_reference_dir
    else:
        reference_dir = os.path.join(
            cfg.val_dataloader.dataset.data_root,
            cfg.val_dataloader.dataset.data_prefix.img_path,
        )

    device = args.device
    dataset = ImagePathDataset(reference_dir)
    if len(dataset) == 0:
        raise FileNotFoundError(f"no .png/.jpg images found under {reference_dir}")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, drop_last=True
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"{len(dataset)} images under {reference_dir} is fewer than "
            f"--batch-size={args.batch_size}; lower --batch-size"
        )

    kid_metric = KernelInceptionDistance(
        subsets=len(dataloader),
        subset_size=args.batch_size,
        reset_real_features=False,
        normalize=False,
    ).to(device)
    with torch.no_grad():
        for batch in dataloader:
            kid_metric.update(batch.to(device).permute(0, 3, 1, 2), real=True)

    ops = args.ops if args.ops else sorted(ALL_DIFFERENTIABLE_PERTURBATIONS)
    unknown = [op for op in ops if op not in ALL_DIFFERENTIABLE_PERTURBATIONS]
    if unknown:
        raise ValueError(
            f"unknown ops {unknown}; choose from {sorted(ALL_DIFFERENTIABLE_PERTURBATIONS)}"
        )

    if args.target_kid is not None:
        target_kid = args.target_kid
        target_desc = f"{target_kid:.6f} (explicit)"
    else:
        ref_op, ref_magnitude = parse_op_at_magnitude(args.target_kid_from)
        target_kid = kid_at(kid_metric, dataloader, ref_op, ref_magnitude, device)
        target_desc = f"{target_kid:.6f} ({ref_op}@{ref_magnitude})"

    grid = np.linspace(0.05, 1.0, args.grid_size)

    print(f"reference images: {reference_dir} ({len(dataset)} images)")
    print(f"target KID: {target_desc}")
    print(f"{'op':<16}{'magnitude':>12}{'kid':>12}")
    magnitudes = {}
    for op_name in ops:
        magnitude, score = calibrate_op(kid_metric, dataloader, op_name, target_kid, grid, device)
        magnitudes[op_name] = {"levels": [magnitude], "probs": [1.0]}
        print(f"{op_name:<16}{magnitude:>12.3f}{score:>12.6f}")

    output_path = args.output or os.path.join(work_dir, "corr_magnitudes_kid_seed.json")
    record = {"iter": 0, "magnitudes": magnitudes}
    with open(output_path, "w") as f:
        json.dump([record], f, indent=2)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
