import os
import json
import warnings
from pprint import pprint
from copy import deepcopy
from typing import Optional, Sequence
from mmengine.visualization import Visualizer

import mmcv
from mmengine.fileio import get
from mmengine.runner import Runner
from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mmseg.structures import SegDataSample
from mmseg.engine.hooks.visualization_hook import SegVisualizationHook

# Local imports
from sensaug.dataset.augmentations import *
from sensaug.sensitivity_analysis import *
from sensaug.runner_utils import *


@HOOKS.register_module()
class AugSegVisualizationHook(SegVisualizationHook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results. Different from the regular SegVisualizationHook because
    the original does not account for dataloader augmentations when visualizing.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def after_test_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
        mode: str = "val",
    ) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

        if self.draw is False or mode == "train":
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for i, output in enumerate(outputs):
                img_path = output.img_path
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order="rgb")
                img_aug = (
                    data_batch["inputs"][i].permute(1, 2, 0).numpy()[..., [2, 1, 0]]
                )
                img_aug = np.ascontiguousarray(
                    img_aug
                )  # needed for viz for whatever reason
                img_aug = cv2.resize(
                    img_aug,
                    (output.gt_sem_seg.shape[1], output.gt_sem_seg.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                # img = [img, img_aug]
                img = [img_aug, img_aug]
                window_name = f"{mode}_{os.path.basename(img_path)}"

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter,
                )


@HOOKS.register_module()
class PerturbationSensitivityAnalysisHook(Hook):
    """Hook for sensitivity analysis, done at start of every epoch."""

    def __init__(self, sa_interval=10000, round_interval=2000) -> None:
        assert round_interval <= sa_interval, (
            "round interval cannot be greater than sa interval"
        )

        self.sa_interval = sa_interval
        self.round_interval = round_interval
        self.sa_curve = None
        self.sa_log_path = None

    def before_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None
    ) -> None:
        # initialize / create the file to log to
        if self.sa_log_path is None:
            self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")
            open(self.sa_log_path, "w+").close()  # clear the file if already exists

        val_dataloader_cfg = deepcopy(runner.cfg.val_dataloader)

        if runner.iter % self.sa_interval == 0:  # perform new sa curve computation
            print("Running sensitivity analysis in before_train_epoch hook.. ")
            self.sa_curve = adaptive_sensitivity_analysis(
                val_dataloader_cfg, runner, num_levels=5, tolerance=0.05
            )  # placeholder

            print("New SA curve computed")
            print(self.sa_curve)

            # SA curve history logging
            sa_curve_str = json.dumps(self.sa_curve)
            with open(self.sa_log_path, "a") as f:
                f.write(sa_curve_str + "\n")

        if (
            runner.iter % self.round_interval == 0
        ):  # evaluate on sa curve and swap datasets if needed
            assert self.sa_curve is not None, "sa curve not computed yet"

            self.levels = {}  # dict to hold the resulting values to train with
            ma_record = {}  # dict to keep track of MA performance of all types and levels

            # iterate through perturbation levels and test their MA performance
            for p_type, levels in self.sa_curve.items():
                ma_record[p_type] = []

                for level in levels:
                    apply_perturbations_dataloader(
                        runner, train=False, perturb_levels={p_type: level}
                    )
                    ma = runner.val()["mIoU"]
                    ma_record[p_type].append(ma)

                self.levels[p_type] = levels[
                    ma_record[p_type].index(min(ma_record[p_type]))
                ]  # get perturb value with worst MA

            print(f"Perturbation values with worst MA: {self.levels}")

            # do some checks
            assert isinstance(val_dataloader_cfg, dict), (
                "train dataloader loaded is not a Dict type"
            )
            assert len(self.levels) == len(PERTURBATIONS), (
                f"need {PERTURBATIONS} values for perturbations in total (6 for RGB, 6 for HSV, blur, noise)"
            )

            # apply_perturbations_dataloader(runner, train=True, perturb_levels=self.levels) # set worst-performing perturbations for train
            create_union_train_set(runner, perturb_levels=self.levels)
            apply_perturbations_dataloader(
                runner, train=False, perturb_levels={}
            )  # reset perturbations to clean for val


@HOOKS.register_module()
class WeightedPerturbationSensitivityAnalysisHook(Hook):
    """Hook for weighted sensitivity analysis, done at start of every epoch."""

    def __init__(self, sa_interval=10000, round_interval=2000) -> None:
        assert round_interval <= sa_interval, (
            "round interval cannot be greater than sa interval"
        )

        self.sa_interval = sa_interval
        self.round_interval = round_interval
        self.sa_curve = None
        self.sa_log_path = None
        self.num_rounds = 0

        self.sa_interval_to_rounds = sa_interval // round_interval

    def update_sa_curve(self, runner, val_dataloader_cfg):
        print("Running sensitivity analysis in before_train_epoch hook...")
        self.sa_curve = adaptive_sensitivity_analysis(
            val_dataloader_cfg, runner, num_levels=5, tolerance=0.05
        )  # placeholder

        print("New SA curve computed")
        print(self.sa_curve)

        # SA curve history logging
        sa_curve_str = json.dumps(self.sa_curve)
        with open(self.sa_log_path, "a") as f:
            f.write(sa_curve_str + "\n")

    def before_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None
    ) -> None:
        # initialize / create the file to log to
        if self.sa_log_path is None:
            self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")

        val_dataloader_cfg = deepcopy(runner.cfg.val_dataloader)
        self.num_rounds = runner.iter // self.round_interval + 1

        if (
            runner.iter % self.round_interval == 0
        ):  # evaluate on sa curve and swap datasets if needed
            # if self.num_rounds % 2 == 0:
            # if self.num_rounds > 1:

            if (
                self.sa_curve is None
                or (self.num_rounds - 1) % self.sa_interval_to_rounds == 0
            ):
                self.update_sa_curve(runner, val_dataloader_cfg)

            assert self.sa_curve is not None, "sa curve not computed yet"

            # dict to keep track of MIOU performance of all types and levels
            miou_record = {}

            # iterate through perturbation levels and test their MIOU performance
            for p_type, levels in self.sa_curve.items():
                miou_record[p_type] = {}

                for level in levels:
                    apply_perturbations_dataloader(
                        runner, train=False, perturb_levels={p_type: level}
                    )
                    miou = runner.val()["mIoU"]
                    miou_record[p_type][level] = miou

            # process miou_record into a probability density function
            pdf_dict = {}
            num_perturbations = len(miou_record.keys())
            perturbation_total_prob = 1.0
            perturbation_uniform_prob = perturbation_total_prob / num_perturbations

            for perturbation, levels in sorted(miou_record.items()):
                weight_sum = sum([(100 - miou) ** 2 for _, miou in levels.items()])

                for level, miou in levels.items():
                    weight = (100 - miou) ** 2
                    pdf_dict[(perturbation, level)] = (
                        weight / weight_sum * perturbation_uniform_prob
                    )

            pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

            # add a "none" perturbation with weight 0.5
            pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

            print("Perturbation PDF computed:")
            print(
                json.dumps(
                    {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
                )
            )

            apply_random_perturbations_train_dataloader(runner, pdf_dict=pdf_dict)
            apply_perturbations_dataloader(
                runner, train=False, perturb_levels={}
            )  # reset perturbations to clean for val

            print("Train Dataloader config:")
            pprint(runner.train_dataloader.dataset.pipeline)

            print("Test Dataloader config:")
            pprint(runner.test_dataloader.dataset.pipeline)

            # else:

            #     apply_perturbations_dataloader(runner, train=True, perturb_levels={})
            #     apply_perturbations_dataloader(runner, train=False, perturb_levels={}) # reset perturbations to clean for val


@HOOKS.register_module()
class PerturbationSensitivityAnalysisHookNew(Hook):
    def __init__(self, sa_interval=10000, round_interval=2000) -> None:
        assert round_interval <= sa_interval, (
            "round interval cannot be greater than sa interval"
        )

        self.sa_interval = sa_interval
        self.round_interval = round_interval
        self.sa_curve = None
        self.sa_log_path = None

    def before_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None
    ) -> None:
        if self.sa_log_path is None:
            self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")
            open(self.sa_log_path, "w+").close()  # clear the file if already exists

        val_dataloader_cfg = deepcopy(runner.cfg.val_dataloader)

        if runner.iter % self.sa_interval == 0:
            print("Running sensitivity analysis in before_train_epoch hook...")
            self.sa_curve = adaptive_sensitivity_analysis_new(
                val_dataloader_cfg, runner, num_levels=5, tolerance=0.05
            )

            print("New SA curve computed")
            print(self.sa_curve)

            sa_curve_str = json.dumps(self.sa_curve)
            with open(self.sa_log_path, "a") as f:
                f.write(sa_curve_str + "\n")

        if runner.iter % self.round_interval == 0:
            assert self.sa_curve is not None, "sa curve not computed yet"

            self.levels = {}
            miou_record = {}

            for p_type, levels in self.sa_curve.items():
                miou_record[p_type] = []

                for level in levels:
                    transform_cls, _ = NEW_PERTURBATIONS[p_type]
                    transform = transform_cls(magnitude=level)
                    apply_perturbations_dataloader_new(
                        runner, train=False, transform=transform
                    )
                    miou = runner.val()["mIoU"]
                    miou_record[p_type].append(miou)

                self.levels[p_type] = levels[
                    miou_record[p_type].index(min(miou_record[p_type]))
                ]

            print(f"Perturbation values with worst mIoU: {self.levels}")

            create_union_train_set_new(runner, perturb_levels=self.levels)
            apply_perturbations_dataloader_new(runner, train=False, transform=None)


@HOOKS.register_module()
class WeightedPerturbationSensitivityAnalysisHookNew(Hook):
    def __init__(self, sa_interval=10000, round_interval=2000) -> None:
        assert round_interval <= sa_interval, (
            "round interval cannot be greater than sa interval"
        )

        self.sa_interval = sa_interval
        self.round_interval = round_interval
        self.sa_curve = None
        self.sa_log_path = None
        self.num_rounds = 0

        self.sa_interval_to_rounds = sa_interval // round_interval

    def update_sa_curve(self, runner, val_dataloader_cfg):
        print("Running sensitivity analysis in before_train_epoch hook...")
        self.sa_curve = adaptive_sensitivity_analysis_new(
            val_dataloader_cfg, runner, num_levels=5, tolerance=0.05
        )

        print("New SA curve computed")
        print(self.sa_curve)

        sa_curve_str = json.dumps(self.sa_curve)
        with open(self.sa_log_path, "a") as f:
            f.write(sa_curve_str + "\n")

    def before_train_iter(
        self, runner: Runner, batch_idx: int, data_batch=None
    ) -> None:
        if self.sa_log_path is None:
            self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")

        val_dataloader_cfg = deepcopy(runner.cfg.val_dataloader)
        self.num_rounds = runner.iter // self.round_interval + 1

        if runner.iter % self.round_interval == 0:
            if (
                self.sa_curve is None
                or (self.num_rounds - 1) % self.sa_interval_to_rounds == 0
            ):
                self.update_sa_curve(runner, val_dataloader_cfg)

            assert self.sa_curve is not None, "sa curve not computed yet"

            miou_record = {}

            for p_type, levels in self.sa_curve.items():
                miou_record[p_type] = {}

                for level in levels:
                    transform_cls, _ = NEW_PERTURBATIONS[p_type]
                    transform = transform_cls(magnitude=level)
                    apply_perturbations_dataloader_new(
                        runner, train=False, transform=transform
                    )
                    miou = runner.val()["mIoU"]
                    miou_record[p_type][level] = miou

            pdf_dict, _ = self.generate_pdf(miou_record)

            apply_random_perturbations_train_dataloader(runner, pdf_dict=pdf_dict)
            apply_perturbations_dataloader_new(runner, train=False, transform=None)

    def generate_pdf(self, miou_record):
        pdf_dict = {}
        num_perturbations = len(miou_record.keys())
        perturbation_total_prob = 1.0
        perturbation_uniform_prob = perturbation_total_prob / num_perturbations

        for perturbation, levels in sorted(miou_record.items()):
            weight_sum = sum([(100 - miou) ** 2 for _, miou in levels.items()])

            for level, miou in levels.items():
                weight = (100 - miou) ** 2
                pdf_dict[(perturbation, level)] = (
                    weight / weight_sum * perturbation_uniform_prob
                )

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

        print("Perturbation PDF computed:")
        print(
            json.dumps(
                {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
            )
        )

        return pdf_dict, None
