from __future__ import print_function
from argparse import ArgumentParser
import contextlib
import os
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
from typing import Dict, List, Union, Tuple

from mmengine.config import Config
from mmengine.runner import Runner
import sensaug.dataset.augmentations  # pylint: disable=unused-import
from library.kid import KID


MAX_COLOR = 1.0
MAX_BLUR = 49
MAX_NOISE = 50
STD_THRESHOLD = 0.01

MASKS = "gtFine/val"
BASE_DATASET = "leftImg8bit/val"
PERTURBATIONS = [
    "lighter_R",
    "lighter_G",
    "lighter_B",
    "darker_R",
    "darker_G",
    "darker_B",
    "lighter_H",
    "lighter_S",
    "lighter_V",
    "darker_H",
    "darker_S",
    "darker_V",
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


def create_gpr_model():
    kernel = RBF() + WhiteKernel()
    return GaussianProcessRegressor(kernel=kernel)


def objective_function(
    level, cfg, kid_calculator, perturbation, ma_base, max_ma, max_kid
):
    """
    Objective function

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
    return abs(ma_base) / abs(ma_base - max_ma) + alpha * (kid_mean / max_kid)


def gpr_optimization(
    cfg, kid_calculator, perturbation, bounds, runner, ma_base, max_ma, max_kid
):
    model = create_gpr_model()
    X_train = np.array([]).reshape(-1, 1)
    y_train = np.array([])

    for i in range(10):
        print(f"{perturbation}: {i + 1} iteration")
        if len(X_train) > 0:
            model.fit(X_train, y_train)

        X_full = np.linspace(bounds[0], bounds[1], 500).reshape(-1, 1)
        y_pred, std_pred = model.predict(X_full, return_std=True)
        if len(X_train) > 0 and max(std_pred) < STD_THRESHOLD:
            print(f"Convergence threshold met for. Stopping optimization.")
            break

        new_point = X_full[np.argmax(std_pred)].reshape(1, -1)
        new_y = objective_function(
            new_point, cfg, kid_calculator, perturbation, ma_base, max_ma, max_kid
        )
        print(f"New sampled point: {new_point.flatten()[0]}, Objective value: {new_y}")

        X_train = np.vstack([X_train, new_point])
        y_train = np.append(y_train, new_y)

    return X_train, y_train


def sensitivity_analysis(cfg: Config, dataset_dir: str) -> Dict[str, List[float]]:
    perturbation_levels = {}
    kid_calculator = KID(os.path.join(dataset_dir, BASE_DATASET), batch_size=25)
    runner = Runner.from_cfg(cfg)
    ma_base = runner.test()["mIoU"]

    for perturbation in PERTURBATIONS:
        print(f"Starting sensitivity analysis for {perturbation}")
        max_ma, max_kid, _ = calculate_ma_kid(
            cfg, kid_calculator, perturbation, MAX_LEVELS[perturbation]
        )
        bounds = (0.1, MAX_LEVELS[perturbation])
        X_train, _ = gpr_optimization(
            cfg, kid_calculator, perturbation, bounds, runner, ma_base, max_ma, max_kid
        )
        perturbation_levels[perturbation] = X_train.flatten().tolist()
        print(f"Completed analysis for {perturbation}")

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
