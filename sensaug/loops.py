import os
import cv2
import json
import logging
from pprint import pprint
from copy import deepcopy

from scipy.stats import betabinom

from mmengine.logging import print_log
from mmengine.device import get_device
from mmengine.runner import ValLoop, TestLoop, IterBasedTrainLoop
from mmseg.registry import LOOPS
from mmengine.dist import is_main_process

# Check Pytorch installation
import torch
from torchmetrics.image.kid import KernelInceptionDistance

# Local imports
# from sensaug.dataset.augmentations import *
from sensaug.sensitivity_analysis import *
from sensaug.runner_utils import *


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


@LOOPS.register_module()
class RobustIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader,
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 1000,
        init_sa=False,
        dynamic_intervals=None,
    ) -> None:
        super().__init__(
            runner, dataloader, max_iters, val_begin, val_interval, dynamic_intervals
        )

        self.init_sa = init_sa

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0 and self._iter < self._max_iters:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data "
                "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        if self.init_sa:
            print_log("Initializing sensitivity analysis curve", logger="current")
            self.runner.val_loop.run()
        else:
            print_log(
                f"Not initializing sensitivity analysis curve. Training will resume at iteration {self._iter}. ",
                logger="current",
            )

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (
                    self._iter % self.val_interval == 0 or self._iter == self._max_iters
                )
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model


@LOOPS.register_module()
class KIDTestLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        assert "corruption_dataloader" in self.runner.cfg.keys(), (
            "corruption dataloader cfg not found"
        )

        self.kid = None
        self.max_iters = int(len(self.dataloader))
        self.update_kid()
        self.kid.reset()  # reset KID metric obj
        assert self.kid is not None, "KID not initialized"

    def update_kid(self):
        self.kid = KernelInceptionDistance(
            subsets=self.max_iters,
            subset_size=8,
            reset_real_features=False,
            normalize=False,
        ).to(get_device())

        # use the corruption dataloader
        dataloader_cfg = deepcopy(self.runner.cfg.corruption_dataloader)

        diff_rank_seed = self.runner._randomness_cfg.get("diff_rank_seed", False)
        corruption_dataloader = self.runner.build_dataloader(
            dataloader_cfg, seed=self.runner.seed, diff_rank_seed=diff_rank_seed
        )

        # update KID with corruption dataloader samples (e.g., one transform from ImageNet-C)
        for idx, data_batch in enumerate(corruption_dataloader):
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=True)

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        self.kid.reset()  # reset KID metric obj

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)

        total_samples = len(self.dataloader.dataset)
        kid_result = self.kid.compute()

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")

        metrics["kid"] = kid_result

        return metrics


@LOOPS.register_module()
class VisualizeSampleLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.img_savedir = os.path.join(runner.work_dir, "img_samples")
        os.makedirs(self.img_savedir, exist_ok=True)

    def run(self) -> dict:
        """Launch test."""

        max_iter = 1

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            if idx >= max_iter:
                break

            batchsize = len(data_batch["inputs"])
            for i in range(batchsize):
                sample_n = idx * batchsize + i
                filename = os.path.join(self.img_savedir, f"sample_{sample_n}.jpg")
                cv2.imwrite(filename, data_batch["inputs"][i].permute(1, 2, 0).numpy())
                print(f"saved to: {filename}")
                break
            # self.run_iter(idx, data_batch)

        # total_samples = int(max_iter * self.dataloader.batch_size)

        # compute metrics
        # metrics = self.evaluator.evaluate(total_samples)
        # self.runner.call_hook('after_test_epoch', metrics=metrics)
        # self.runner.call_hook('after_test')

        # return metrics


@LOOPS.register_module()
class DebugAugLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(self, runner, dataloader, evaluator, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.img_savedir = os.path.join(runner.work_dir, "img_samples")
        os.makedirs(self.img_savedir, exist_ok=True)

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        perturbation_list = list(NEW_PERTURBATIONS.items())

        magnitudes = [0.25, 0.50, 0.75, 1.0]
        sample = None

        for pname, _ in perturbation_list:
            for m in magnitudes:
                apply_perturbations_dataloader(
                    self.runner, train=False, perturb_levels={pname: m}
                )
                print(f"== pname: {pname} || magnitude: {m}")

                for idx, data_batch in enumerate(self.dataloader):
                    sample = data_batch
                    break

                filename = os.path.join(self.img_savedir, f"sample_{pname}_{m}.jpg")
                cv2.imwrite(filename, sample["inputs"][0].permute(1, 2, 0).numpy())
                print(f"saved to: {filename}")


@LOOPS.register_module()
class SubsetValLoop(ValLoop):
    """Custom loop for test, uses a subset."""

    def __init__(
        self, runner, dataloader, evaluator, ratio=0.5, fp16: bool = False
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.ratio = ratio
        self.dataloader_iter = iter(self.dataloader)

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        max_iter = int(len(self.dataloader) * self.ratio)

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics


@LOOPS.register_module()
class SubsetTestLoop(TestLoop):
    """Custom loop for test, uses a subset."""

    def __init__(
        self, runner, dataloader, evaluator, ratio=0.5, fp16: bool = False
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        self.ratio = ratio
        # self.root_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kid = None
        self.update_kid()

    def update_kid(self):
        max_iter = int(len(self.dataloader) * self.ratio)
        # max_iter = int(len(self.dataloader))

        self.kid = KernelInceptionDistance(
            subsets=max_iter,
            subset_size=4,
            reset_real_features=False,
            normalize=False,
        ).to(get_device())

        # set test performance on
        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=True)
            idx += 1

        self.kid.reset()  # reset KID metric obj

    def run(self) -> dict:
        """Launch test."""

        max_iter = int(len(self.dataloader) * self.ratio)

        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()

        self.kid.reset()  # reset KID metric obj

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)
        kid_result = self.kid.compute()

        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")

        metrics["kid"] = kid_result

        return metrics

    def run_eval_only(self) -> dict:
        """Launch test."""

        max_iter = int(len(self.dataloader) * self.ratio)
        self.runner.model.eval()
        self.kid.reset()  # reset KID metric obj

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            data_batch_inputs = torch.stack(data_batch["inputs"], dim=0).to(
                get_device()
            )
            self.kid.update(data_batch_inputs, real=False)
            idx += 1

        total_samples = int(max_iter * self.dataloader.batch_size)
        kid_result = self.kid.compute()
        # compute metrics
        metrics = self.evaluator.evaluate(total_samples)
        metrics["kid"] = kid_result

        return metrics


@LOOPS.register_module()
class RobustValLoop(ValLoop):
    """Custom loop for val."""

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio=0.5,
        sa_curve_path=None,
        uniform=False,
        descending_MA=False,
        remove_H=False,
        warmup_rounds: int = 4,
        random_aug: bool = False,
        geometric_only: bool = False,
        photometric_only: bool = False,
        weighted_augs: bool = False,
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)

        assert sa_curve_path is not None, (
            "sa curve path must be provided for evaluation"
        )

        self.ratio = ratio
        self.kid = None
        self.sa_curve = None
        self.pdf_dict = None
        self.remove_H = remove_H
        self.uniform = uniform
        self.warmup_rounds: int = warmup_rounds
        self.descending_MA = descending_MA
        self.static_sa_curve_path = sa_curve_path
        self.sa_log_path = os.path.join(runner.cfg.work_dir, "sa_curve_log.txt")
        self.perturb_metrics_path = os.path.join(
            runner.cfg.work_dir, "perturb_eval.txt"
        )
        self.n_rounds = runner.iter // runner.cfg.train_cfg.val_interval
        self.random_aug = random_aug
        self.geometric_only = geometric_only
        self.photometric_only = photometric_only
        self.weighted_augs = weighted_augs

        with open(sa_curve_path, "r") as f:
            sa_curve_str = f.read()
            self.eval_sa_curve = json.loads(sa_curve_str)

        if is_main_process():
            print(
                f"Starting from round {self.n_rounds}; \
                Uniform sampling: {self.uniform}; \
                Descending MA: {self.descending_MA}; \
                Remove color augs: {self.remove_H}; \
                Warmup rounds: {self.warmup_rounds}"
            )

    def load_sa_curve(self):
        assert self.static_sa_curve_path is not None, "No SA curve path provided"
        path = self.static_sa_curve_path
        with open(path, "r") as f:
            sa_curve_str = f.read()
            self.sa_curve = json.loads(sa_curve_str)

    def update_sa_curve(self):
        # if get_rank() == 0:
        print_log("Running sensitivity analysis...", logger="current")
        val_dataloader_cfg = deepcopy(self.runner.cfg.val_dataloader)
        self.sa_curve = adaptive_sensitivity_analysis_new(
            val_dataloader_cfg, self.runner, num_levels=5, tolerance=0.05
        )

        assert self.sa_curve is not None, "SA curve is None after broadcasting"

        if is_main_process():
            print("New SA curve computed")
            print(self.sa_curve)

            # SA curve history logging
            sa_curve_str = json.dumps(self.sa_curve)
            with open(self.sa_log_path, "a") as f:
                f.write(sa_curve_str + "\n")

    def test_perturbed_new(self):
        # dict to keep track of MIOU performance of all types and levels
        miou_record = {}
        metrics_record = {}
        final_metrics = {}

        # iterate through perturbation levels and test their MIOU performance
        for p_type, levels in self.sa_curve.items():
            miou_record[p_type] = {}
            metrics_record[p_type] = []

            for _, level in enumerate(levels):
                # evaluate on current SA curve to choose PDF
                apply_perturbations_dataloader(
                    self.runner, train=False, perturb_levels={p_type: level}
                )
                metrics = self._run_eval(ratio=self.ratio)
                miou = metrics["mIoU"]
                miou_record[p_type][level] = miou

                metrics_eval = metrics
                metrics_record[p_type].append(metrics_eval)

            metrics_record[p_type] = dict_mean(
                metrics_record[p_type]
            )  # average metric for p type across all levels

        for p_type, mean_dict in metrics_record.items():
            for metric, value in mean_dict.items():
                new_key = p_type.replace("_", "") + f"_{metric}"
                final_metrics[new_key] = value

        return miou_record, final_metrics

    def test_perturbed(self):
        # dict to keep track of MIOU performance of all types and levels
        miou_record = {}
        metrics_record = {}
        final_metrics = {}

        # iterate through perturbation levels and test their MIOU performance
        for p_type, levels in self.sa_curve.items():
            miou_record[p_type] = {}
            metrics_record[p_type] = []

            for i, level in enumerate(levels):
                # evaluate on current SA curve to choose PDF
                apply_perturbations_dataloader(
                    self.runner, train=False, perturb_levels={p_type: level}
                )
                metrics = self._run_eval(ratio=self.ratio)
                miou = metrics["mIoU"]
                miou_record[p_type][level] = miou

                metrics_eval = metrics
                metrics_record[p_type].append(metrics_eval)

            metrics_record[p_type] = dict_mean(
                metrics_record[p_type]
            )  # average metric for p type across all levels

        for p_type, mean_dict in metrics_record.items():
            for metric, value in mean_dict.items():
                new_key = p_type.replace("_", "") + f"_{metric}"
                final_metrics[new_key] = value

        return miou_record, final_metrics

    def generate_uniform_pdf(self):
        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        # remove H perturbations from training
        if self.remove_H:
            miou_record.pop("PosterizeTransform", None)
            miou_record.pop("SolarizeTransform", None)

        # process miou_record into a probability density function
        pdf_dict = {}
        num_perturbations = len(miou_record.keys()) + 1  # add 1 for "none" perturbation

        for perturbation, levels in sorted(miou_record.items()):
            num_levels = len(levels.keys())
            for level, _ in levels.items():
                pdf_dict[(perturbation, level)] = 1.0 / (num_perturbations * num_levels)

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation to pdf
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob
        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    def generate_pdf_new_weighted_aug(self):
        """same as generate_pdf_new, but we dont give uniformity to augmentation classes---they are all weighted by MA."""

        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        # remove H perturbations from training
        if self.remove_H:
            miou_record.pop("PosterizeTransform", None)
            miou_record.pop("SolarizeTransform", None)

        # process miou_record into a probability density function
        pdf_dict = {}
        reverse = not self.descending_MA

        all_mious = {}
        for perturbation, levels in sorted(miou_record.items()):
            for i, (level, miou) in enumerate(
                sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
            ):
                all_mious[(perturbation, level)] = miou

        for i, ((perturbation, level), miou) in enumerate(
            sorted(all_mious.items(), key=lambda x: x[1], reverse=reverse)
        ):

            def lambda_bb(x):
                return betabinom.pmf(x, len(all_mious), 0.75, 1.0)

            pdf_dict[(perturbation, level)] = lambda_bb(i) * 0.95

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

        if is_main_process():
            print("Perturbation PDF computed:")
            print(
                json.dumps(
                    {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
                )
            )

        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    def generate_pdf_new(self):
        assert self.sa_curve is not None, "SA curve is None in RobustValLoop"
        miou_record, final_metrics = self.test_perturbed_new()

        # remove H perturbations from training
        if self.remove_H:
            miou_record.pop("PosterizeTransform", None)
            miou_record.pop("SolarizeTransform", None)

        # process miou_record into a probability density function
        pdf_dict = {}
        num_perturbations = len(miou_record.keys())
        perturbation_total_prob = num_perturbations / float(num_perturbations + 1)
        perturbation_uniform_prob = perturbation_total_prob / num_perturbations

        for perturbation, levels in sorted(miou_record.items()):

            def lambda_bb(x):
                return betabinom.pmf(x, len(levels), 0.75, 1.0)

            # iterate through order of decreasing miou
            reverse = not self.descending_MA
            for i, (level, _) in enumerate(
                sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
            ):
                pdf_dict[(perturbation, level)] = (
                    lambda_bb(i) * perturbation_uniform_prob
                )

        pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

        # add a "none" perturbation
        pdf_dict[("none", 0)] = 1.0 - pdf_dict_perturb_prob

        if is_main_process():
            print("Perturbation PDF computed:")
            print(
                json.dumps(
                    {str(k): v for k, v in pdf_dict.items()}, indent=4, sort_keys=True
                )
            )

        self.pdf_dict = pdf_dict

        return pdf_dict, final_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook("before_val_iter", batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with torch.autocast(device_type="cuda", enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            "after_val_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs
        )

    def _run_eval(self, ratio):
        max_iter = int(len(self.dataloader) * ratio)
        n_samples = 0

        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            n_samples += len(data_batch["inputs"])
            idx += 1

        # compute metrics
        metrics = self.evaluator.evaluate(n_samples)
        return metrics

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        # perturb_metrics = self._run_eval(ratio=1.0)
        perturb_metrics = {}

        if self.n_rounds >= self.warmup_rounds:
            if self.random_aug:
                apply_random_alpha_training_augmentations(  # noqa: F405
                    self.runner,
                    geometric_only=self.geometric_only,
                    photometric_only=self.photometric_only,
                )

            else:
                if (self.n_rounds - self.warmup_rounds) % 6 == 0:
                    self.update_sa_curve()

                if self.sa_curve is not None:
                    _, perturb_metrics = (
                        self.generate_uniform_pdf()
                        if self.uniform
                        else (
                            self.generate_pdf_new_weighted_aug()
                            if self.weighted_augs
                            else self.generate_pdf_new()
                        )
                    )  # update pdf to sample from
                    apply_random_perturbations_train_dataloader_new(  # noqa: F405
                        self.runner, pdf_dict=self.pdf_dict
                    )  # type: ignore

        # full clean evaluation last
        apply_perturbations_dataloader(self.runner, train=False, perturb_levels={})
        metrics = self._run_eval(ratio=1.0)

        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)

            # perturb eval logging
            metrics_str = json.dumps(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(metrics_str + "\n")

            print("Train Dataloader config:")
            pprint(self.runner.train_dataloader.dataset.pipeline)

            print("Test Dataloader config:")
            pprint(self.runner.test_dataloader.dataset.pipeline)

        # we log the clean evaluation officially to monitor progress
        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1

        return perturb_metrics


@LOOPS.register_module()
class RobustBaselineValLoop(RobustValLoop):
    """Custom loop for val."""

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio=0.5,
        sa_curve_path=None,
        fp16: bool = False,
    ) -> None:
        super().__init__(
            runner,
            dataloader,
            evaluator,
            ratio=ratio,
            sa_curve_path=sa_curve_path,
            fp16=fp16,
        )

        self.load_sa_curve()

    def run(self) -> dict:
        """Launch test."""

        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()

        _, perturb_metrics = self.test_perturbed()

        # full clean evaluation last
        apply_perturbations_dataloader(self.runner, train=False, perturb_levels={})
        metrics = self._run_eval(ratio=1.0)

        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)

            # perturb eval logging
            metrics_str = json.dumps(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(metrics_str + "\n")

            print("Train Dataloader config:")
            pprint(self.runner.train_dataloader.dataset.pipeline)

            print("Test Dataloader config:")
            pprint(self.runner.test_dataloader.dataset.pipeline)

        # we log the clean evaluation officially to monitor progress
        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1

        return perturb_metrics
