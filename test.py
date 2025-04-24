# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from tools/test.py
import argparse
import os
import glob
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import VISUALIZERS
from mmengine.visualization import Visualizer
import sensaug.dataset.datasets as datasets
from sensaug.hooks import *
from sensaug.visualizer import BPSegLocalVisualizer
from sensaug.dataset.idbh import IDBHTransform  # noqa:F401

from SEG_CONFIG import DATA_ROOT_LOOKUP

PROXY_ALPHAS = {
    "voc12": [0.6245, 0.6056, 0.6558, [0.2058, 1.8083], 0.0009],
    "cityscapes": [0.5984, 0.6247, 0.5537, [0.9921, 0.3702], 0.0057],
    "potsdam": [0.5320, 0.4758, 0.4792, [2.8966, 0.2955], 0.0066],
    "cub": [-0.3199, 0.4667, 0.6306, [0.4772, 0.2014], 0.0005],
    "loveda": [0.5193, 0.4332, 0.3107, [0.7910, 0.6551], 0.0004],
}


def apply_dark_zurich_eval(cfg, args, original_work_dir):
    cfg.dataset_type = "DarkZurichDataset"
    cfg.data_root = (
        "/fs/nexus-projects/robustness_datasets/segmentation/dark_zurich_val"
    )
    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path="rgb_anon/val/night/GOPR0356", seg_map_path="gt/val/night/GOPR0356"
    )

    cfg.test_dataloader = cfg.val_dataloader

    cfg = cfg_switch_work_dir(cfg, args, os.path.join(original_work_dir, "dark_zurich"))

    return cfg


def apply_nighttime_driving_eval(cfg, args, original_work_dir):
    cfg.dataset_type = "NightDrivingDataset"
    cfg.data_root = (
        "/fs/nexus-projects/robustness_datasets/segmentation/NighttimeDrivingTest"
    )
    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path="leftImg8bit/test/night",
        seg_map_path="gtCoarse_daytime_trainvaltest/test/night",
    )

    cfg.test_dataloader = cfg.val_dataloader

    cfg = cfg_switch_work_dir(
        cfg, args, os.path.join(original_work_dir, "nightdriving")
    )

    return cfg


def apply_acdc_eval(cfg, args, original_work_dir, split="all"):
    cfg.dataset_type = "ACDCDataset"
    cfg.data_root = DATA_ROOT_LOOKUP["acdc"]
    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root

    if split == "all":
        split = ""
    else:
        split = "_" + split

    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path=f"rgb_anno{split}/test", seg_map_path=f"gt{split}/test"
    )
    cfg.test_dataloader = cfg.val_dataloader
    cfg = cfg_switch_work_dir(
        cfg, args, os.path.join(original_work_dir, f"acdc{split}")
    )

    return cfg


def apply_idd_eval(cfg, args, original_work_dir):
    cfg.dataset_type = "IDDDataset"
    cfg.data_root = DATA_ROOT_LOOKUP["idd"]
    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path="leftImg8bit/val", seg_map_path="gtFine/val"
    )
    cfg.test_dataloader = cfg.val_dataloader

    cfg = cfg_switch_work_dir(cfg, args, os.path.join(original_work_dir, "idd"))

    return cfg


def cfg_switch_work_dir(cfg, args, path):
    cfg.work_dir = path

    if args.show or args.save_vis:
        cfg = trigger_visualization_hook(cfg, args)

    if args.out is not None:
        cfg.test_evaluator["output_dir"] = os.path.join(path, args.out)
        cfg.test_evaluator["keep_results"] = True

    return cfg


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
            "if specified, the evaluation metric results will be dumpedinto the directory as json"
        ),
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        default=False,
        help="test all models in workdir",
    )
    parser.add_argument(
        "--test_data",
        default=None,
        choices=["acdc", "idd", "zurich", "night"],
        help="test datasets; overrides test dataset inferred from config. default: None",
    )
    parser.add_argument(
        "--use-best",
        action="store_true",
        default=True,
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
            "if specified, the evaluation metric results will be dumpedinto the directory as json"
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
        help="If specified, visualization will be automatically saved to the work_dir",
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
    cfg["visualizer"] = dict(
        type="BPSegLocalVisualizer",
        alpha=0.4,
        vis_backends=[dict(type="LocalVisBackend")],
        name="visualizer",
    )

    default_hooks = cfg.default_hooks

    if "visualization" in default_hooks:
        default_hooks.pop("visualization")  # remove visualization default hook

        # Turn on visualization
        visualization_hook = dict(type="AugSegVisualizationHook")
        visualization_hook["draw"] = True
        visualization_hook["interval"] = 2
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.save_vis:
            visualizer = cfg.visualizer
            visualizer["save_dir"] = os.path.join(cfg.work_dir)
            print(f"Save dir changed to: {visualizer['save_dir']}")
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    cfg.default_hooks["visualization"] = visualization_hook
    return cfg


def main():
    args = parse_args()

    workdir_list = glob.glob(os.path.join(args.work_dir, "*/"))

    if args.test_all:
        for workdir in workdir_list:
            if (
                "eval" not in workdir
                and "loveda" in workdir
                and os.path.isdir(workdir[:-1] + "_eval")
            ):
                args.work_dir = workdir
                print("testing on workdir: ", workdir)

                try:
                    test_once(args)
                except Exception as e:
                    print("test failed for: ", workdir)
                    print("Error msg: ")
                    print(e)
                    pass

    else:
        print("testing on workdir: ", args.work_dir)
        test_once(args)


def test_once(args):
    cfg, original_work_dir = build_cfg(args)

    if args.test_data is None:
        print(f"Testing on data specified by config: {cfg.dataset_type}")
        runner = Runner.from_cfg(cfg)
        runner.test()
        runner.visualizer.__class__._instance_dict.clear()
        Visualizer._instance_dict.clear()
        print(f"Testing done! Workdir: {cfg.work_dir}")

    elif args.test_data == "zurich":
        print("Testing on dark zurich.. ")
        cfg = apply_dark_zurich_eval(cfg, args, cfg.work_dir)
        runner = Runner.from_cfg(cfg)
        runner.test()
        runner.visualizer.__class__._instance_dict.clear()
        Visualizer._instance_dict.clear()
        print(f"Dark Zurich testing done! Workdir: {cfg.work_dir}")

    elif args.test_data == "night":
        print("Testing on nighttime driving.. ")
        cfg = apply_nighttime_driving_eval(cfg, args, cfg.work_dir)
        runner = Runner.from_cfg(cfg)
        runner.test()
        runner.visualizer.__class__._instance_dict.clear()
        Visualizer._instance_dict.clear()
        print(f"Nighttime Driving testing done! Workdir: {cfg.work_dir}")

    elif args.test_data == "acdc":
        splits = ["all", "fog", "night", "rain", "snow"]

        original_work_dir = cfg.work_dir

        for s in splits:
            print(f"Testing on ACDC, split {s} .. ")
            cfg = apply_acdc_eval(cfg, args, original_work_dir, split=s)
            runner = Runner.from_cfg(cfg)
            runner.test()
            runner.visualizer.__class__._instance_dict.clear()
            Visualizer._instance_dict.clear()
            print(f"ACDC testing done for split: {s}! Workdir: {cfg.work_dir}")

    elif args.test_data == "idd":
        print("Testing on IDD.. ")
        cfg = apply_idd_eval(cfg, args, cfg.work_dir)
        runner = Runner.from_cfg(cfg)
        runner.test()
        runner.visualizer.__class__._instance_dict.clear()
        Visualizer._instance_dict.clear()
        print(f"IDD testing done! Workdir: {cfg.work_dir}")


def build_cfg(args):
    work_dir_exists = os.path.isdir(args.work_dir)
    if work_dir_exists:
        config_path = glob.glob(os.path.join(args.work_dir, "*.py"))[0]
        args.config = config_path
        cfg = Config.fromfile(config_path)
    else:
        # load config
        cfg = Config.fromfile(args.config)

    original_work_dir = cfg.work_dir
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
            # cfg.resume = True

        print(f"Loading checkpoint path: {last_checkpoint_path}")

        cfg.load_from = last_checkpoint_path
        args.pretrained = False  # pretrained is false no matter what... for now

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        basename_dir = os.path.basename(os.path.normpath(args.work_dir))
        parent_dir = os.path.dirname(os.path.normpath(args.work_dir)) + "_eval"
        cfg.work_dir = (
            os.path.join(parent_dir, basename_dir) if work_dir_exists else args.work_dir
        )
        print(f"cfg work dir set to: {cfg.work_dir}")

    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    if args.data_root is not None:
        print(f"Setting data root: {args.data_root}")
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

    # cfg['visualizer']= dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend'),
    #                     dict(type='TensorboardVisBackend')])

    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    cfg.custom_hooks = None
    cfg.resume = False
    cfg.default_hooks.logger.interval = 5

    return cfg, original_work_dir


if __name__ == "__main__":
    main()
