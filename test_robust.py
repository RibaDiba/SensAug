import argparse
import os
import gc
import glob
import os.path as osp
from pprint import pprint
import json
from copy import deepcopy
from typing import Dict, List

from mmengine.logging import print_log
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from sensaug.dataset.augmentations import (
    FastRGBTransform,
    FastBlurTransform,
    FastNoiseTransform,
    NEW_PERTURBATIONS,
    IMAGENETC_NAME_FN_DICT,
)
from sensaug.analysis import eval_results_to_csv
from sensaug.sensitivity_analysis import adaptive_sensitivity_analysis
from sensaug.runner_utils import (
    apply_perturbations_dataloader,
)
from sensaug.loops import (
    SubsetTestLoop,
    DebugAugLoop,
)  # import required; don't delete
from sensaug.dataset.datasets import ACDCDataset
from sensaug.hooks import AugSegVisualizationHook
from sensaug.visualizer import BPSegLocalVisualizer
from sensaug.dataset.idbh import IDBHTransform # noqa:F401


def main():
    args = parse_args()
    test_once(args)


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("--config", default=None, help="train config file path")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint file",
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument(
        "--use-best",
        action="store_true",
        default=False,
        help="use the best mIoU checkpoint instead of the latest checkpoint",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="load pretrained weights",
    )
    parser.add_argument(
        "--exp_name",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Data location, if not the same as the provided config",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        default=False,
        help="If specified, visualization will be automatically saved "
        "to the work_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    palette = None

    if "synapse" in args.work_dir:
        palette = [
            [0, 0, 0],  # 'background'
            [0, 0, 255],  # aorta
            [0, 255, 0],  # gallbladder
            [255, 0, 0],  # left_kidney
            [0, 255, 255],  # right_kidney
            [255, 0, 255],  # liver
            [255, 255, 0],  # pancreas
            [60, 255, 255],  # spleen
            [243, 117, 43],  # stomach, orange
        ] + [[0, 0, 0] for _ in range(10)]
        assert len(palette) == 19

    cfg["visualizer"] = dict(
        type="BPSegLocalVisualizer",
        alpha=0.75,
        palette=palette,
        vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
        name="visualizer",
    )

    default_hooks = cfg.default_hooks

    if "visualization" in default_hooks:
        default_hooks.pop("visualization")  # remove visualization default hook

        # Turn on visualization
        visualization_hook = dict(type="AugSegVisualizationHook")
        visualization_hook["draw"] = True
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.save_vis:
            visualizer = cfg.visualizer
            visualizer["save_dir"] = os.path.join(cfg.work_dir)
            print_log(
                f"Save dir changed to: {visualizer['save_dir']}", logger="current"
            )
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    cfg.default_hooks["visualization"] = visualization_hook

    return cfg


def cfg_switch_work_dir(cfg, args, path):
    cfg.work_dir = path
    os.makedirs(cfg.work_dir, exist_ok=True)

    if args.show or args.save_vis:
        cfg = trigger_visualization_hook(cfg, args)

    if args.out is not None:
        cfg.test_evaluator["output_dir"] = os.path.join(path, args.out)
        cfg.test_evaluator["keep_results"] = True

    return cfg


def test_robust(cfg, args, sa_results_file="sensaug/testing/test_levels.json"):
    perturb_levels = {}

    with open(sa_results_file, "r") as fp:
        perturb_levels = json.load(fp)

    # add imagenetc transforms
    perturb_levels["motion_blur"] = list(range(1, 6))
    perturb_levels["zoom_blur"] = list(range(1, 6))
    perturb_levels["pixelate"] = list(range(1, 6))
    perturb_levels["jpeg_compression"] = list(range(1, 6))
    perturb_levels["snow"] = list(range(1, 6))
    perturb_levels["frost"] = list(range(1, 6))
    perturb_levels["fog"] = list(range(1, 6))

    # add combination transforms
    perturb_levels["combination"] = list(range(0, 6))

    original_work_dir = cfg.work_dir

    print("Clean Evaluation.. ")
    cfg = cfg_switch_work_dir(cfg, args, os.path.join(original_work_dir, "eval_clean"))
    runner = Runner.from_cfg(cfg)
    runner.test()  # test clean data

    # remove current visualizer (necessary to not overwrite files)
    runner.visualizer.__class__._instance_dict.clear()
    Visualizer._instance_dict.clear()
    del runner
    gc.collect()

    for p_type, levels in perturb_levels.items():
        for i, level in enumerate(levels):
            results_file_list = glob.glob(
                os.path.join(
                    original_work_dir, f"eval_ptype={p_type}_level={i+1}", "*", "*.json"
                )
            )

            if len(results_file_list) > 0:
                print_log(
                    f"Skipping {p_type} level {i+1} because it already exists",
                    logger="current",
                )
                continue

            print_log(f"{p_type} Evaluation at level {i+1}.. ", logger="current")
            cfg = cfg_switch_work_dir(
                cfg,
                args,
                os.path.join(original_work_dir, f"eval_ptype={p_type}_level={i+1}"),
            )
            runner = Runner.from_cfg(cfg)

            # construct the dataloader here
            apply_perturbations_dataloader(
                runner, train=False, perturb_levels={p_type: level}
            )

            # run evaluation
            runner.test()

            # remove current visualizer (necessary to not overwrite files)
            runner.visualizer.__class__._instance_dict.clear()
            Visualizer._instance_dict.clear()
            del runner
            gc.collect()

    eval_results_to_csv(original_work_dir)
    print_log("Perturbed evaluation done.", logger="current")


def visualize_dataloader(cfg, args):
    cfg.test_cfg.type = "DebugAugLoop"
    cfg["visualizer"] = dict(
        type="SegLocalVisualizer",
        vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
    )

    runner = Runner.from_cfg(cfg)
    runner.test()

    print_log(f"Visualization done. Saved to: {args.work_dir}", logger="current")


def test_once(args):
    args.use_best = True

    work_dir_exists = os.path.isdir(args.work_dir)
    if work_dir_exists:
        config_path = glob.glob(os.path.join(args.work_dir, "*.py"))[0]
        args.config = config_path
        cfg = Config.fromfile(config_path)
    else:
        # load config
        cfg = Config.fromfile(args.config)

    cfg.test_cfg = dict(type="TestLoop")  # Could be SubsetTestLoop before
    cfg.test_evaluator = cfg.val_evaluator
    cfg.test_evaluator["collect_device"] = "gpu"
    cfg.val_cfg = dict(type="ValLoop")

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.checkpoint is not None:
        cfg.load_from = args.checkpoint
        args.pretrained = False

    elif work_dir_exists:
        if args.use_best:  # use the best checkpoint
            last_checkpoint_path = glob.glob(os.path.join(args.work_dir, "best*.pth"))
            assert len(last_checkpoint_path) > 0, "No best checkpoint found"
            last_checkpoint_path = last_checkpoint_path[0]
        else:  # use latest checkpoint
            last_checkpoint_f = os.path.join(args.work_dir, "last_checkpoint")

            with open(last_checkpoint_f, "r") as file:
                last_checkpoint_path = os.path.basename(file.read().strip())

            last_checkpoint_path = os.path.join(args.work_dir, last_checkpoint_path)
            cfg.resume = True

        print_log(f"Loading checkpoint path: {last_checkpoint_path}", logger="current")

        cfg.load_from = last_checkpoint_path
        args.pretrained = False  # pretrained is false no matter what... for now

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:  # put eval in a different directory..
        # update configs according to CLI args if args.work_dir is not None
        basename_dir = os.path.basename(os.path.normpath(args.work_dir))
        parent_dir = os.path.dirname(os.path.normpath(args.work_dir)) + "_eval"
        cfg.work_dir = (
            os.path.join(parent_dir, basename_dir) if work_dir_exists else args.work_dir
        )

    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    if args.data_root is not None:
        print_log(f"Setting data root: {args.data_root}", logger="current")
        cfg.train_dataloader.dataset.data_root = args.data_root
        cfg.test_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.data_root = args.data_root

    if args.show or args.save_vis:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.out = args.out
        cfg.test_evaluator["output_dir"] = os.path.join(cfg.work_dir, args.out)
        cfg.test_evaluator["keep_results"] = True
        cfg.val_evaluator["output_dir"] = os.path.join(cfg.work_dir, args.out)
        cfg.val_evaluator["keep_results"] = True

    # cfg['randomness'] = dict(seed=0)
    cfg.test_dataloader = cfg.val_dataloader

    # test
    # cfg.work_dir = cfg.work_dir + "_TEMP"
    test_levels = "sensaug/testing/test_levels_new.json"
    print_log(f"Testing on levels: {test_levels}", logger="current")
    test_robust(cfg, args, sa_results_file=test_levels)


if __name__ == "__main__":
    main()
