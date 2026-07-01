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
class PerturbationSensitivityAnalysisHookWithGradients(Hook): 

    """
    custom hook to intigrate the graident calculations into the loop
    """

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

    def calculate_cross_corelation(self): 
        """function to compute cross corelation"""
        pass

    def prune_augmentations(self):
        """takes a look at the augmentation list and prunes augmnentations that are correlated"""
        pass

    def generate_pdf(self, miou_record):
        """this needs to be chanegd """
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
