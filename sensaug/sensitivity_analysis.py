"""Script for running sensitivity analysis on base image dataset."""

from __future__ import print_function
from typing import Callable, Dict, List, Tuple

import os
import json
from copy import deepcopy
from argparse import ArgumentParser
import bisect
import itertools

from pprint import pprint
from time import time

from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.runner import Runner
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sensaug.dataset.augmentations import *  # PERTURBATIONS, NEW_PERTURBATIONS, GEOMETRIC_TRANSFORMS
from sensaug.runner_utils import apply_perturbations_dataloader

# Cityscapes dataset (https://www.cityscapes-dataset.com/downloads/):
# - gtFine_trainvaltest.zip
# - leftImg8bit_trainvaltest.zip

# after installing cityscapes, flatten the directory structure by removing cities

# pip install cityscapesscripts
# python library/mmsegmentation-1.2.1/tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8

# mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .


# dataset
MASKS = "gtFine/val"
BASE_DATASET = "leftImg8bit/val"


def adaptive_sensitivity_analysis_new(cfg, runner, num_levels, tolerance):
    predictor = pchip_interpolator
    perturbation_levels: Dict[str, List[float]] = {}
    apply_perturbations_dataloader(runner, train=False, perturb_levels={})

    print_log(f"{int(time())}: Running baseline performance test", logger="current")
    miou_base: float = runner.test_loop.run()["mIoU"]

    min_level = 0.0
    max_level = 1.0

    perturbation_list = NEW_PERTURBATIONS.items()

    if hasattr(cfg, "geometric_only"):
        perturbation_list = NEW_PERTURBATIONS_GEOMETRIC.items()

    if hasattr(cfg, "photometric_only"):
        perturbation_list = NEW_PERTURBATIONS_PHOTOMETRIC.items()

    for perturbation, _ in perturbation_list:
        error = tolerance * max_level

        # calculate mean IoU and KID for baseline and max perturbation
        max_miou, max_kid, _ = calculate_miou_kid_new(
            cfg, runner, perturbation, max_level
        )
        points = np.array([(min_level, 0.0), (max_level, 2.0)])

        # target y values that are evenly spaced between 0 and 2
        targets = np.linspace(0, 2, num_levels + 1)[1:-1]

        levels, errors = predictor(points, targets)

        # check if max error is less then tolerance
        index = np.argmax(errors)
        level, max_error = levels[index], errors[index]

        while max_error >= error:
            # calculate mean IoU and KID for perturbation level
            miou, kid, _ = calculate_miou_kid_new(cfg, runner, perturbation, level)
            value = objective_function(
                level, max_level, miou_base - miou, miou_base - max_miou, kid, max_kid
            )
            points = np.insert(
                points, bisect.bisect(points[:, 0], level), (level, value), axis=0
            )

            levels, errors = predictor(points, targets)

            # check if max error is less then tolerance
            index = np.argmax(errors)
            level, max_error = levels[index], errors[index]

        # final perturbation levels
        final_levels = list(levels)
        # final_levels.append(max_level)
        perturbation_levels[perturbation] = final_levels

        print(f"{int(time())}: Sampled {len(points) - 1} levels")
        print(f"{int(time())}: Final perturbation levels: {final_levels}")

    apply_perturbations_dataloader(runner, train=False, perturb_levels={})
    return perturbation_levels


def adaptive_sensitivity_analysis(
    cfg: Config,
    runner: Runner,
    num_levels: int,
    tolerance: float,
    predictor: Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]] = None,
) -> Dict[str, List[float]]:
    """Adaptively find perturbation levels with denser sampling around sensitive levels.

    Args:
        cfg: Model config
        runner: Model runner
        num_levels: Number of perturbation levels to find
        tolerance: Maximum error tolerance
        predictor: Function that predicts x values at y values given starting points

    Returns:
        Dictionary of perturbations and best levels
    """
    if predictor is None:
        predictor = pchip_interpolator

    # setup
    perturbation_levels: Dict[str, List[float]] = {}
    reset_dataloader_runner(runner)

    print_log(f"{int(time())}: Running baseline performance test", logger="current")
    miou_base: float = runner.test_loop.run()["mIoU"]

    # iterate over each perturbation
    for perturbation, max_level in PERTURBATIONS.items():
        print_log(
            f"{int(time())}: Finding best perturbation levels for {perturbation}...",
            logger="current",
        )

        error = tolerance * max_level
        if perturbation == "blur":
            error = max(1.0, error)

        # calculate mean IoU and KID for baseline and max perturbation
        max_miou, max_kid, _ = calculate_miou_kid(cfg, runner, perturbation, max_level)
        points = np.array([(0.0, 0.0), (max_level, 2.0)])

        # target y values that are evenly spaced between 0 and 2
        targets = np.linspace(0, 2, num_levels + 1)[1:-1]

        # adaptively find perturbation levels
        while True:
            # estimate best perturbation levels
            levels, errors = predictor(points, targets)

            if perturbation == "blur":
                levels = 2.0 * np.floor(levels / 2) + 1.0

            # check if max error is less then tolerance
            index = np.argmax(errors)
            level, max_error = levels[index], errors[index]
            if max_error < error:
                break

            # calculate mean IoU and KID for perturbation level
            miou, kid, _ = calculate_miou_kid(cfg, runner, perturbation, level)
            value = objective_function(
                level, max_level, miou_base - miou, miou_base - max_miou, kid, max_kid
            )
            points = np.insert(
                points, bisect.bisect(points[:, 0], level), (level, value), axis=0
            )

        # final perturbation levels
        final_levels = list(levels)
        # final_levels.append(max_level)
        if perturbation == "blur":
            final_levels = [int(i) for i in final_levels]

        perturbation_levels[perturbation] = final_levels

        print(f"{int(time())}: Sampled {len(points) - 1} levels")
        print(f"{int(time())}: Final perturbation levels: {final_levels}")

    reset_dataloader_runner(runner)

    print("Reset val / test dataloader:")
    pprint(runner.test_dataloader.dataset.pipeline)

    return perturbation_levels


def uniform_sensitivity_analysis(
    cfg: Config, runner: Runner, num_levels: int, sample_levels: int
) -> Dict[str, List[float]]:
    """Calculate ground truth values for each perturbation and level.

    Args:
        cfg: Model config
        runner: Model runner
        num_levels: Number of perturbation levels to find
        sample_levels: Number of perturbation levels to sample

    Returns:
        Dictionary of perturbations and best levels
    """
    # setup
    perturbation_levels: Dict[str, List[float]] = {}
    reset_dataloader_runner(runner)

    print(f"{int(time())}: Running baseline performance test")
    ma_base: float = runner.test_loop.run()["mIoU"]

    # iterate over each perturbation
    for perturbation, max_level in PERTURBATIONS.items():
        print(f"{int(time())}: Finding best perturbation levels for {perturbation}...")

        # uniformly sample perturbation levels
        levels: NDArray[np.float32] = np.linspace(0, max_level, sample_levels + 1)[1:]

        if perturbation == "blur":
            levels = 2.0 * np.floor(levels / 2) + 1.0

        points: List[Tuple[float, float, float]] = []
        for level in levels:
            miou, kid, _ = calculate_miou_kid(cfg, runner, perturbation, level)
            points.append((level, ma_base - miou, kid))

        # find best perturbation levels
        max_level, max_miou, max_kid = points.pop()
        final_levels: List[float] = []
        maximum = -np.inf
        for combination in itertools.combinations(points, num_levels - 1):
            values = [
                objective_function(level, max_level, miou, max_miou, kid, max_kid)
                for level, miou, kid in combination
            ]
            values = [0] + values + [2]

            # find minimum difference between values
            value = min(values[i + 1] - values[i] for i in range(len(values) - 1))
            if value > maximum:
                final_levels = [point[0] for point in combination]
                maximum = value

        # final_levels.append(max_level)
        if perturbation == "blur":
            final_levels = [int(i) for i in final_levels]

        perturbation_levels[perturbation] = final_levels

    reset_dataloader_runner(runner)

    return perturbation_levels


def reset_dataloader_runner(runner: Runner) -> None:
    """Reset dataloader for runner with config.

    Args:
        cfg: Model config
        runner: Model runner
    """
    apply_perturbations_dataloader(runner, train=False, perturb_levels={})


def objective_function(
    level: float,
    max_level: float,
    miou: float,
    max_miou: float,
    kid: float,
    max_kid: float,
) -> float:
    """Objective function that represents cumulative sensitivity.

    Args:
        level: Perturbation level
        max_level: Maximum perturbation level
        miou: Mean IoU
        max_miou: Maximum mean IoU
        kid: KID
        max_kid: Maximum KID

    Returns:
        Objective value
    """
    return (
        (miou / (max_miou + np.finfo(np.float32).eps))
        + 2 * (level / max_level)
        - (kid / (max_kid + np.finfo(np.float32).eps))
    )


def calculate_miou_kid_new(cfg, runner, perturbation, level):
    apply_perturbations_dataloader(
        runner, train=False, perturb_levels={perturbation: level}
    )

    # Assumes Test loop is of type SubsetTestLoop (train_robust.py)
    assert runner.cfg.test_cfg.type == "SubsetTestLoop", (
        f"Test loop expected to be of type SubsetTestLoop but is {runner.cfg.test_cfg.type}"
    )

    metrics = runner.test_loop.run()
    miou = metrics["mIoU"]
    kid = metrics.get("kid", (np.nan, np.nan))
    kid_mean, kid_std = kid
    ##

    print(
        f"{perturbation} {level}: {miou:.3f}, KID mean: {kid_mean:.3f}, KID std: {kid_std:.3f}"
    )
    return miou, float(kid_mean), float(kid_std)


def calculate_miou_kid(
    cfg: Config, runner: Runner, perturbation: str, level: float
) -> Tuple[float, float, float]:
    """Calculate mean IoU and KID.

    Args:
        cfg: Model config
        kid: KID calculator
        perturbation: Perturbation name
        level: Perturbation level

    Returns:
        mIoU
        KID mean
        KID standard deviation
    """
    if perturbation == "blur":
        transform = dict(type="GaussianBlurPerturbation", kernel_size=int(level))
    elif perturbation == "noise":
        transform = dict(type="GaussianNoisePerturbation", sigma=level)
    elif perturbation == "distort":
        transform = dict(type="RadialDistortionPerturbation", level=level)
    else:
        direction_str, channel_str = perturbation.split("_")
        direction = 1 if direction_str == "lighter" else 0

        if channel_str in "BGR":
            channel = "BGR".index(channel_str)
            name = "RGBPerturbation"
        else:
            channel = "HSV".index(channel_str)
            name = "HSVPerturbation"

        transform = dict(type=name, channel=channel, alpha=level, direction=direction)

    cfg.dataset.pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        transform,
        dict(type="PackSegInputs"),
    ]

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )
    runner.test_loop.dataloader = new_dataloader
    runner.val_loop.dataloader = new_dataloader

    # Assumes Test loop is of type SubsetTestLoop (train_robust.py)
    metrics = runner.test_loop.run()
    miou = metrics["mIoU"]
    kid_mean, kid_std = metrics["kid"]

    print(f"{perturbation} {level}: {miou}, {kid_mean}, {kid_std}")

    return miou, kid_mean.item(), kid_std.item()


def pchip_interpolator(points: NDArray, targets: NDArray) -> Tuple[NDArray, NDArray]:
    """Estimate function values and errors at target points using PCHIP interpolation.

    Args:
        points: Array of points
        targets: Array of target y values

    Returns:
        Function values
        Errors
    """
    values: List[float] = []
    errors: List[float] = []

    for target in targets:
        interpolator = PchipInterpolator(points[:, 0], points[:, 1], extrapolate=False)
        value = interpolator.solve(target)[0]

        # check if level is already in points
        if np.any(np.abs(points[:, 0] - value) < 1e-10):
            values.append(value)
            errors.append(0)
            continue

        # calculate possible x and y differences
        index = bisect.bisect(points[:, 0], value) - 1
        x_diff = (points[index + 1, 0] - points[index, 0]) / points[-1, 0]
        y_diff = min(
            abs(interpolator(points[index, 0]) - target),
            abs(interpolator(points[index + 1, 0]) - target),
        )

        # lower bound interpolation
        points1 = np.insert(
            points, index + 1, (value, target - y_diff * (0.6 + x_diff / 5)), axis=0
        )
        value1 = PchipInterpolator(
            points1[:, 0], points1[:, 1], extrapolate=False
        ).solve(target)[0]

        # upper bound interpolation
        points2 = np.insert(
            points, index + 1, (value, target + y_diff * (0.6 + x_diff / 5)), axis=0
        )
        value2 = PchipInterpolator(
            points2[:, 0], points2[:, 1], extrapolate=False
        ).solve(target)[0]

        values.append(value)
        errors.append(abs(value1 - value2) / 2)

    return np.array(values), np.array(errors)


def gaussian_process_regressor(
    points: NDArray, targets: NDArray
) -> Tuple[NDArray, NDArray]:
    """Estimate function values and errors at target points using PCHIP interpolation.

    Args:
        points: Array of points
        targets: Array of target y values

    Returns:
        Function values
        Errors
    """
    kernel = RBF() + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(points[:, 0], points[:, 1])
    return model.predict(targets, return_std=True)


def main():
    parser = ArgumentParser(
        prog="Basis Perturbation Sensitivity Analysis",
        description='This script outputs perturbation parameters that are the "furthest apart".',
    )

    parser.add_argument("--dataset", required=True, help="path of input dataset")
    parser.add_argument("--scratch", required=True, help="path of scratch directory")
    parser.add_argument("--config", required=True, help="filename of model config")
    parser.add_argument("--model", required=True, help="filename of model checkpoint")
    parser.add_argument(
        "--type",
        default="adaptive",
        choices=["adaptive", "uniform"],
        help="type of sensitivity analysis",
    )
    parser.add_argument("--points", default=5, help="number of levels to find")
    parser.add_argument("--error", default=0.05, help="maximum error (for adaptive)")
    parser.add_argument("--samples", default=10, help="number of samples (for uniform)")
    parser.add_argument(
        "--use_new_sa",
        action="store_true",
        help="Use the new sensitivity analysis functions",
    )

    args = parser.parse_args()

    print(f"{int(time())}: Conducting sensitivity analysis...")

    # load config
    config = Config.fromfile(os.path.join(args.scratch, args.config))

    config.work_dir = os.path.join(args.scratch, "work_dirs")
    config.load_from = os.path.join(args.scratch, args.model)
    config.data_root = args.dataset
    config.test_dataloader.dataset.data_prefix = {
        "img_path": BASE_DATASET,
        "seg_map_path": MASKS,
    }
    config.test_dataloader.dataset.data_root = args.dataset
    config.test_dataloader.dataset.pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="PackSegInputs"),
    ]

    # build the runner from config
    runner = Runner.from_cfg(config)

    # run sensitivity analysis
    if args.use_new_sa:
        if args.type == "adaptive":
            perturbation_levels = adaptive_sensitivity_analysis_new(
                config, runner, args.points, args.error
            )
        else:
            raise ValueError(
                "Uniform sensitivity analysis is not implemented for the new method."
            )
    else:
        if args.type == "adaptive":
            perturbation_levels = adaptive_sensitivity_analysis(
                config, runner, args.points, args.error, pchip_interpolator
            )
        else:
            perturbation_levels = uniform_sensitivity_analysis(
                config, runner, args.points, args.samples
            )

    print(f"{int(time())}: Done conducting sensitivity analysis!")

    # save the perturbation levels
    savefile = os.path.join(args.scratch, "perturbation_levels.json")
    with open(savefile, "w+", encoding="utf-8") as fp:
        json.dump(perturbation_levels, fp, indent=4)

    print(f"Saved perturbation levels to: {savefile}")
