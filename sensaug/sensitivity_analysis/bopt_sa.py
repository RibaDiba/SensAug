from __future__ import print_function
from typing import Dict, List, Tuple, Union
from argparse import ArgumentParser
import contextlib
import os
import time

from mmengine.config import Config
from mmengine.runner import Runner
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import sensaug.dataset.augmentations  # pylint: disable=unused-import
from library.kid import KID
import numpy as np

MAX_COLOR = 1.0
MAX_BLUR = 49
MAX_NOISE = 50
MAX_DISTORT = 500
EI_THRESHOLD = 0.0001

# dataset
MASKS = "gtFine/val"
BASE_DATASET = "leftImg8bit/val"

PERTURBATIONS = [
    # "lighter_R",
    # "lighter_G",
    # "lighter_B",
    # "darker_R",
    # "darker_G",
    # "darker_B",
    # "lighter_H",
    # "lighter_S",
    # "lighter_V",
    # "darker_H",
    # "darker_S",
    # "darker_V",
    "blur",
    "noise",
    # "distort"
]
MAX_LEVELS = {
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
    # "distort": MAX_DISTORT
}
alpha = 1.0


# Objective function for Bayesian optimization
def objective_function(
    level, cfg, kid_calculator, perturbation, ma_base, max_ma, max_kid
):
    """
    Objective function to minimize in Bayesian optimization.

    Args:
        level: Perturbation level to evaluate.
        cfg: Configuration for the model.
        kid_calculator: KID calculator instance.
        perturbation: Type of perturbation.
        ma_base: Baseline mean accuracy.
        max_ma: Maximum mean accuracy.
        max_kid: Maximum KID value.

    Returns:
        Combined objective value based on mean accuracy and KID.
    """
    ma, kid_mean, _ = calculate_ma_kid(cfg, kid_calculator, perturbation, level)
    return abs(ma_base - ma) / abs(ma_base - max_ma) + alpha * (kid_mean / max_kid)


# Main function for Bayesian optimization
def bayesian_optimization(
    cfg, kid_calculator, perturbation, bounds, runner, ma_base, max_ma, max_kid
):
    """
    Perform Bayesian optimization to find optimal perturbation levels.
    """
    space = [Real(bounds[0], bounds[1], name="level")]
    print(f"Starting Bayesian optimization for {perturbation}...")

    @use_named_args(space)
    def objective(level):
        print(f"Evaluating {perturbation} at level {level}")
        obj_value = objective_function(
            level, cfg, kid_calculator, perturbation, ma_base, max_ma, max_kid
        )
        print(f"Objective Value = {obj_value}")
        return obj_value

    start_time = time.time()

    res_gp = gp_minimize(objective, space, n_calls=8, n_random_starts=3)

    for i in range(8, 20):
        x_next = res_gp.space.transform(
            [
                res_gp.x_iters[
                    np.argmax(
                        gaussian_ei(
                            res_gp.space, res_gp.models[-1], xi=res_gp.x_iters[-1]
                        )
                    )
                ]
            ]
        )
        ei = gaussian_ei(x_next, res_gp.models[-1], xi=res_gp.x_iters[-1])

        if ei < EI_THRESHOLD:
            print("Expected Improvement below threshold. Stopping optimization.")
            break

        res_gp = gp_minimize(
            objective,
            space,
            n_calls=1,
            x0=res_gp.x_iters,
            y0=res_gp.func_vals,
            n_random_starts=0,
        )

    end_time = time.time()
    plot_convergence(res_gp)
    print(
        f"Optimization for {perturbation} completed in {end_time - start_time} seconds. Levels: {res_gp.x_iters}"
    )
    return res_gp.x_iters, res_gp.func_vals


def sensitivity_analysis(cfg: Config, dataset_dir: str) -> Dict[str, List[float]]:
    """
    Conducts sensitivity analysis using Bayesian optimization.

    Args:
        cfg: Configuration for the model.
        dataset_dir: Path to the dataset directory.

    Returns:
        Dictionary of perturbations and their best perturbation levels.
    """

    perturbation_levels: Dict[str, List[float]] = {}
    kid_calculator = KID(os.path.join(dataset_dir, BASE_DATASET), batch_size=25)
    runner = Runner.from_cfg(cfg)

    ma_base = runner.test()["mIoU"]

    for perturbation in PERTURBATIONS:
        max_ma, max_kid, _ = calculate_ma_kid(
            cfg, kid_calculator, perturbation, MAX_LEVELS[perturbation]
        )
        bounds = (0.1, MAX_LEVELS[perturbation])
        levels, _ = bayesian_optimization(
            cfg, kid_calculator, perturbation, bounds, runner, ma_base, max_ma, max_kid
        )
        perturbation_levels[perturbation] = sorted(
            set([level for sublist in levels for level in sublist])
        )
        print(f"Final levels for {perturbation}: {perturbation_levels[perturbation]}")

    return perturbation_levels


def calculate_ma_kid(
    cfg: Config, kid: KID, perturbation: str, level: Union[int, float]
) -> Tuple[float, float, float]:
    """Calculate mean accuracy and KID.

    Args:
        cfg: Model config
        kid: KID calculator
        perturbation: Perturbation name
        level: Perturbation level

    Returns:
        mIoU Mean accuracy
        KID mean
        KID standard deviation
    """
    if perturbation == "blur":
        transform = dict(type="GaussianBlurPerturbation", kernel_size=level)
    elif perturbation == "noise":
        transform = dict(type="GaussianNoisePerturbation", sigma=level)
    else:
        direction_str, channel_str = perturbation.split("_")
        direction = 1 if direction_str == "lighter" else 0

        if channel_str in "RGB":
            channel = "RGB".index(channel_str)
            name = "RGBPerturbation"
        else:
            channel = "HSV".index(channel_str)
            name = "HSVPerturbation"

        transform = dict(type=name, channel=channel, alpha=level, direction=direction)

    cfg.test_dataloader.dataset.pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        transform,
        dict(type="PackSegInputs"),
    ]

    # build runner and suppress output
    with open(os.devnull, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        runner = Runner.from_cfg(cfg)

    ma = runner.test()["mIoU"]
    kid_mean, kid_std = kid.compute(perturbation, level)

    return ma, kid_mean.item(), kid_std.item()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Basis Perturbation Sensitivity Analysis",
        description='This script outputs perturbation parameters that are the "furthest apart".',
    )

    parser.add_argument("--dataset", required=True, help="path of input dataset")
    parser.add_argument("--scratch", required=True, help="path of scratch directory")
    parser.add_argument("--config", required=True, help="filename of model config")
    parser.add_argument("--model", required=True, help="filename of model checkpoint")
    parser.add_argument("--points", default=5, help="number of levels to find")
    parser.add_argument("--error", default=0.01, help="maximum error")

    args = parser.parse_args()

    # load config
    config = Config.fromfile(os.path.join(args.scratch, args.config))

    config.work_dir = os.path.join(args.scratch, "work_dir")
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

    sensitivity_analysis(config, args.dataset)

    print("Sensitivity analysis completed")
