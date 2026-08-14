import os
import numpy as np
import re
import glob
import logging
import shutil

from sensaug.cluster_config import load_seg_config

from mmengine.logging import print_log
import mmseg
import mmpretrain.models  # noqa:F401

from mmengine import Config
from mmengine.hooks import EarlyStoppingHook  # noqa:F401
from mmengine.runner import Runner
from mmengine.dist import is_main_process

# Check Pytorch installation
import torch

import sensaug.dataset.datasets as datasets  # noqa:F401
from sensaug.dataset.augmentations import *  # noqa:F403
from sensaug.dataset.idbh import IDBHTransform  # noqa:F401
from sensaug.dataset.vip import VIPAugTransform  # noqa:F401
from sensaug.hooks import *  # noqa:F403
from sensaug.loops import *  # noqa:F403
from sensaug.visualizer import BPSegLocalVisualizer  # noqa:F401

def dist_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# from torch.utils.data import DataLoader
dist_print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
dist_print(mmseg.__version__)


def atoi(text):
    return int(text) if text.isdigit() else text


def trigger_visualization_hook(cfg, args):
    cfg["visualizer"] = dict(
        type="BPSegLocalVisualizer",
        alpha=0.4,
        vis_backends=[dict(type="TensorboardVisBackend")],
        name="visualizer",
    )
    # dict(type='LocalVisBackend'),

    default_hooks = cfg.default_hooks

    if "visualization" in default_hooks:
        default_hooks.pop("visualization")  # remove visualization default hook

        # Turn on visualization
        visualization_hook: dict = dict(type="AugSegVisualizationHook")
        visualization_hook["draw"] = True
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


def resolve_interval(cli_value, key, default):
    """Resolve one pipeline's clock: CLI flag > cluster config `schedule:` > default.

    All three are iteration counts. The checks are explicit `is not None` rather
    than `or`, so a deliberate 0 reaches the hook that validates it and fails
    loudly, instead of silently falling through to the default.
    """
    if cli_value is not None:
        return cli_value
    configured = SCHEDULE.get(key)
    return default if configured is None else configured


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def apply_acdc_train_eval(cfg, split="all"):
    cfg.dataset_type = "ACDCDataset"
    cfg.data_root = DATA_ROOT_LOOKUP["acdc"]

    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = DATA_ROOT_LOOKUP["acdc"]

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = DATA_ROOT_LOOKUP["acdc"]

    if split == "all":
        split = ""
    else:
        split = "_" + split

    cfg.train_dataloader.dataset.data_prefix = dict(
        img_path=f"rgb_anno{split}/train", seg_map_path=f"gt{split}/train"
    )
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path=f"rgb_anno{split}/test", seg_map_path=f"gt{split}/test"
    )
    cfg.test_dataloader = cfg.val_dataloader

    return cfg


def _localize_pretrained_checkpoint(cfg, backbone):
    """Redirect backbone.init_cfg checkpoint URLs to PRETRAINED_CACHE_DIR.

    Compute nodes on Della have no internet access, so any config whose backbone
    init_cfg points at a download.openmmlab.com URL (segformer, pspnet-rsb,
    convnext, swin) crashes deep inside model.init_weights() with a DNS error.
    Fails fast here instead, before Runner/NCCL/dataloaders spin up, if the local
    copy hasn't been staged yet via scripts/download_pretrained_checkpoints.py.
    """
    if not PRETRAINED_CACHE_DIR or "model" not in cfg:
        return

    def _walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "Pretrained" and str(
            node.get("checkpoint", "")
        ).startswith("http"):
            url = node["checkpoint"]
            local_path = os.path.join(PRETRAINED_CACHE_DIR, os.path.basename(url))
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"Pretrained checkpoint not cached locally: {url}\n"
                    f"Expected at: {local_path}\n"
                    "Compute nodes have no internet access -- fetch it first from a "
                    "login node (or your machine + rsync) with:\n"
                    f"  python scripts/download_pretrained_checkpoints.py --backbone {backbone}"
                )
            node["checkpoint"] = local_path
            return
        for v in node.values():
            _walk(v)

    _walk(cfg.model)


def build_config(args):
    if args.use_foundation_backbone and args.dataset == "cityscapes":
        config_path = os.path.dirname(MMCONFIG_PATH)
        print(f"{config_path}/vitsam_cityscapes_1024.py")
        cfg = Config.fromfile(f"{config_path}/vitsam_cityscapes_1024.py")
    else:
        if "acdc" in args.dataset:  # use cityscapes config for acdc
            mm_configs = glob.glob(f"{MMCONFIG_PATH}/{args.backbone}/*cityscapes*.py")
        else:
            mm_configs = glob.glob(
                f"{MMCONFIG_PATH}/{args.backbone}/*{args.dataset}*.py"
            )
        mm_configs.sort(
            key=natural_keys
        )  # sorting to use the smallest resnet backbone available

        print("Existing configs found: ", mm_configs)

        if len(mm_configs) == 0:  # no configs exist for this configuration
            mm_configs = glob.glob(
                f"{MMCONFIG_PATH}/{args.backbone}/*.py"
            )  # use any existing config
            mm_configs.sort(key=natural_keys)
            config = mm_configs.pop(0)

            while (
                "r101" in config and len(mm_configs) > 0
            ):  # don't use a big model if we don't have to, lol
                config = mm_configs.pop(0)

            dist_print(f"Using base config: {config}")
            cfg = Config.fromfile(config)

            # update dataset
            dataset_name = args.dataset
            data_cfg = Config.fromfile(
                f"{MMCONFIG_PATH}/_base_/datasets/{dataset_name}.py"
            ).to_dict()
            cfg.merge_from_dict(data_cfg)

            # update schedule
            original_optim_wrapper = cfg.get("optim_wrapper", None)
            schedule_cfg = Config.fromfile(
                f"{MMCONFIG_PATH}/_base_/schedules/schedule_320k.py"
            ).to_dict()
            cfg.merge_from_dict(schedule_cfg)
            if original_optim_wrapper is not None:
                cfg.optim_wrapper = original_optim_wrapper
                cfg.optimizer = original_optim_wrapper.optimizer

        else:
            config = mm_configs[0]
            dist_print(f"Using existing base config: {config}")
            cfg = Config.fromfile(config)

    cfg.launcher = args.launcher
    cfg.pretrained = None
    cfg.model.pretrained = None
    _localize_pretrained_checkpoint(cfg, args.backbone)

    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                f"`--amp` is only supported when the optimizer wrapper type is `OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # enable automatically scaling LR
    if "auto_scale_lr" in cfg and "base_batch_size" in cfg.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    cfg.test_dataloader = cfg.val_dataloader

    # Set up working dir to save files and logs.
    cfg.work_dir = os.path.join(args.work_dir, args.exp_name)

    if (
        args.resume
        or os.path.isdir(cfg.work_dir)
        and os.path.isfile(os.path.join(cfg.work_dir, "last_checkpoint"))
    ):
        cfg.resume = True
        cfg.load_from = None

    cfg.default_hooks.checkpoint.interval = cfg.train_cfg.max_iters // 50
    cfg.default_hooks.checkpoint.save_best = PRIMARY_METRIC
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3

    # grad_corr with the SA loop turned off trains exactly like `none`: the
    # correlation pipeline is a measurement, and with no SA there is nothing to
    # drive a training augmentation pdf. This is the control arm -- an R measured
    # against an unaugmented baseline is what the SA-on number gets compared to.
    plain_pipeline = args.aug_type == "none" or (
        args.aug_type == "grad_corr" and args.no_corr_sa
    )

    if plain_pipeline:  # use no augmentations
        excluded_augmentations = [
            "PhotoMetricDistortion",
            "RandomFlip",
            "RandomResize",
            "PackSegInputs",
        ]
        pipeline = [
            x
            for x in cfg.train_dataloader.dataset.pipeline
            if (x["type"] not in excluded_augmentations)
        ]
        pipeline.append(dict(type="PackSegInputs"))

    else:  # use custom augmentations
        excluded_augmentations = [
            "PhotoMetricDistortion",
            "RandomFlip",
            "RandomResize",
            "PackSegInputs",
            "LoadAnnotations",
        ]
        pipeline = [
            x
            for x in cfg.train_dataloader.dataset.pipeline
            if (x["type"] not in excluded_augmentations)
        ]

        augmentation_type = args.aug_type
        if augmentation_type == "autoaugment":
            pipeline.append(dict(type="AutoAugmentTransform"))
        elif augmentation_type == "augmix":
            pipeline.append(dict(type="AugMixTransform"))
        elif augmentation_type == "randaugment":
            pipeline.append(dict(type="RandAugmentTransform"))
        elif augmentation_type == "trivialaugment":
            pipeline.append(dict(type="TrivialAugmentWideTransform"))
        elif augmentation_type in ("random", "ours", "grad_corr", "default"):
            pipeline.append(
                dict(
                    type="RandomAlphaTrainTransform",
                    geometric_only=args.geometric_only,
                    photometric_only=args.photometric_only,
                    # grad_corr/ours/default all train on "aligned": the same 32 ops
                    # R is computed over, played through the plain CPU transform
                    # classes. That is what lets a per-op score read off R index
                    # straight into the training pdf -- "new" is 20 PascalCase names
                    # against R's 32 snake_case ones, and they overlap on nothing.
                    # Keeping all three of these arms on the same vocabulary is also
                    # what makes them comparable to each other in the first place --
                    # they're meant to differ in how they pick/weight from the set,
                    # not in which ops are available. "random" is left on "new" on
                    # purpose: it isn't one of the compared arms.
                    #
                    # NOT "diff", which has the right names but the wrong
                    # implementation: those ops are GPU-batched-only by design (see
                    # differentiable_augmentations.py) and 40-150x slower applied
                    # per-image on CPU, which is what this pipeline does.
                    perturbation_set=(
                        "aligned"
                        if augmentation_type in ("grad_corr", "ours", "default")
                        else "new"
                    ),
                )
            )
        elif augmentation_type == "idbh":
            pipeline.append(dict(type="IDBHTransform", version="cifar10-weak"))
        elif augmentation_type == "vip":
            # kernel=2, vital=options['vital'], nonvital=options['nonvital'], dataroot=options['data'], dataroot_c=options['data_c'], num_workers=options['workers'], batch_size=options['batch_size'], _transforms=options['aug'], _eval=options['eval'], fractal_images=options['fractal_path']
            pipeline.append(
                dict(
                    type="VIPAugTransform",
                    kernel=2,
                    vital=0.001,
                    nonvital=0.005,
                    dataset_name="cityscapes",
                    fractal_images="./sensaug/dataset/vip_fractals/images_224_tiny/",
                )
            )

        pipeline.append(dict(type="PackSegInputs"))
        pipeline.insert(1, dict(type="LoadAnnotations"))

    cfg.train_pipeline = pipeline
    cfg.train_dataloader.dataset.pipeline = pipeline

    # Set data root
    data_root = DATA_ROOT_LOOKUP[args.dataset]
    dist_print(f"Setting data root: {data_root}")
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root

    # set up visualizer
    cfg.randomness = dict(seed=0)
    np.random.seed(0)
    cfg.visualizer = dict(
        type="Visualizer", vis_backends=[dict(type="TensorboardVisBackend")]
    )

    N_ROUNDS = 20
    N_CORR_EMISSIONS = 4

    # The two pipelines' clocks, resolved independently of each other. round_interval
    # drives the SA pipeline (RobustValLoop; its SA-curve recompute is every 6th of
    # these rounds, in sensaug/loops.py). corr_interval drives the gradient
    # cross-correlation pipeline. Neither is derived from the other.
    round_interval = resolve_interval(
        args.round_interval, "round_interval", cfg.train_cfg.max_iters // N_ROUNDS
    )
    corr_interval = resolve_interval(
        args.corr_interval, "corr_interval", cfg.train_cfg.max_iters // N_CORR_EMISSIONS
    )
    cfg.train_cfg.val_interval = round_interval

    # if "acdc" not in args.dataset.lower():
    #     cfg.train_cfg.max_iters = (
    #         cfg.train_cfg.max_iters * 2
    #     )  # NOTE: since we have early stopping, we just increase this.

    cfg.default_hooks.logger.interval = 200
    cfg.default_hooks.checkpoint.interval = cfg.train_cfg.max_iters // 20
    cfg.default_hooks.checkpoint.save_best = "mIoU"
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3

    # for i in range(len(cfg.param_scheduler)):
    #     cfg.param_scheduler[i].begin *= 2
    #     cfg.param_scheduler[i].end *= 2

    cfg.test_evaluator = cfg.val_evaluator

    args.save_vis = True
    args.show = False

    # grad_corr reuses the SA machinery, but over the DIFFERENTIABLE vocabulary so
    # the SA curve and the matrix R are keyed by the same augmentations. Disabled
    # with --no-corr-sa, which leaves the stock val loop and probes at the fixed
    # reference magnitude.
    sa_loop = args.aug_type == "ours" or (
        args.aug_type == "grad_corr" and not args.no_corr_sa
    )

    if sa_loop:
        eval_ratio = 0.25 if "acdc" not in args.dataset.lower() else 1.0
        cfg.val_cfg.type = "RobustValLoop"
        # "aligned", not "diff": same 32 op names either way, but the CPU classes
        # rather than the per-image torch wrappers. The SA round-eval inserts these
        # into the val dataloader once per (op, level) pair, and on the diff
        # vocabulary that was the dominant cost of a grad_corr run. ours also trains
        # on "aligned" now (see the pipeline-construction block above), so its SA
        # phase must probe the same vocabulary its uniform warmup phase used.
        cfg.val_cfg.perturbation_set = (
            "aligned" if args.aug_type in ("grad_corr", "ours") else "new"
        )
        cfg.val_cfg.ratio = eval_ratio
        cfg.val_cfg.sa_curve_path = "sensaug/testing/test_levels_voc.json"
        cfg.val_cfg.uniform = args.uniform
        # cfg.val_cfg.uniform = False
        cfg.val_cfg.descending_MA = args.descending_MA  # defaults to False
        # cfg.val_cfg.descending_MA = False # NOTE: False --> severe augmentations prioritized in pdf
        cfg.val_cfg.remove_H = args.no_inv_aug
        cfg.val_cfg.warmup_rounds = 0 if args.no_warmup else 4
        cfg.val_cfg.random_aug = args.random_aug
        cfg.val_cfg.geometric_only = args.geometric_only
        cfg.val_cfg.photometric_only = args.photometric_only
        cfg.val_cfg.weighted_augs = args.weighted_augs
        cfg.val_cfg.corr_lambda = args.corr_lambda
        cfg.val_cfg.corr_lambda_ramp = args.corr_lambda_ramp
        # cfg.val_cfg.remove_H = False
        cfg.test_cfg.type = "SubsetTestLoop"
        cfg.test_cfg.ratio = eval_ratio
        cfg.train_cfg.type = "RobustIterBasedTrainLoop"
        cfg.train_cfg.init_sa = True if (cfg.resume or args.no_warmup) else False

        # NOTE: LoveDA doesn't converge very well on default lr; we should reduce it by 1 order mag
        if "loveda" in args.dataset.lower():
            cfg.optimizer.lr *= 0.1

    if args.aug_type == "grad_corr":
        # Gradient-based augmentation cross-correlation. Every `emit_interval`
        # iters CollectGradientHook freezes the model and sweeps the whole clean
        # val set for d loss / d magnitude, then
        # PerturbationSensitivityAnalysisHookWithGradients correlates that sweep
        # into R. Both gate on the SAME interval, so they are built from one
        # variable rather than two that could drift apart.
        #
        # --corr-sync-sa just hands them the SA loop's clock instead of their own.
        # No special gate is needed: fires_at() counts runner.iter + 1, which is the
        # value IterBasedTrainLoop tests against val_interval right after this hook
        # point, so passing round_interval lands the sweep on exactly the iterations
        # that are SA rounds.
        #
        # ORDERING, and it matters: RobustIterBasedTrainLoop calls val_loop.run()
        # AFTER run_iter, so a synced sweep fires BEFORE the SA round it is synced
        # to updates the pdf. It therefore probes at the previous round's
        # magnitudes -- which is the right semantics (those are the magnitudes that
        # were in effect over the window being measured) but is not obvious.
        emit_interval = round_interval if args.corr_sync_sa else corr_interval

        # The priorities are load-bearing, not cosmetic: both hooks act in
        # after_train_iter, and the correlation hook must see the sweep the
        # collector just wrote. NORMAL (50) runs before LOW (70).
        cfg.custom_hooks = (cfg.get("custom_hooks") or []) + [
            dict(
                type="CollectGradientHook",
                interval=emit_interval,
                sweep_batch_size=1,
                magnitude_mode=args.corr_magnitude_mode,
                magnitudes_path=args.corr_magnitudes,
                priority="NORMAL",
            ),
            dict(
                type="PerturbationSensitivityAnalysisHookWithGradients",
                interval=emit_interval,
                red_mode=args.corr_red_mode,
                mask_within_op=not args.corr_keep_within_op,
                priority="LOW",
            ),
        ]

    if args.adamw:
        lr = cfg.optimizer.lr
        cfg.optimizer = dict(type="AdamW", lr=lr, weight_decay=0.0005)
        cfg.optim_wrapper.optimizer = cfg.optimizer

    if args.freeze_early_layers:  # freeze first 8 layers of 12 total
        cfg.model.backbone.frozen_stages = 8  # type: ignore

    cfg.geometric_only = args.geometric_only
    cfg.photometric_only = args.photometric_only

    cfg.randomness = dict(seed=0, diff_rank_seed=False)

    # Let's have a look at the final config used for training
    dist_print(f"Config:\n{cfg.pretty_text}")
    return cfg


def set_manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train(args):
    cfg = build_config(args)
    os.makedirs(cfg.work_dir, exist_ok=True)
    shutil.copy(args.cluster_config, os.path.join(cfg.work_dir, "seg_config.yaml"))
    runner = Runner.from_cfg(cfg)
    set_manual_seed(0)  # set seed
    runner.val_loop  # initialize val loop
    runner.test_loop
    runner.train()


if __name__ == "__main__":
    import argparse

    # Two-stage parse: load cluster config first so SUPPORTED_* are available for choices
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--cluster-config", required=True)
    _pre_args, _ = _pre.parse_known_args()
    _seg = load_seg_config(_pre_args.cluster_config)
    MMCONFIG_PATH       = _seg["MMCONFIG_PATH"]
    PRIMARY_METRIC      = _seg["PRIMARY_METRIC"]
    DATA_ROOT_LOOKUP    = _seg["DATA_ROOT_LOOKUP"]
    SUPPORTED_DATASETS  = _seg["SUPPORTED_DATASETS"]
    SUPPORTED_BACKBONES = _seg["SUPPORTED_BACKBONES"]
    SCHEDULE            = _seg["SCHEDULE"]
    PRETRAINED_CACHE_DIR = _seg["PRETRAINED_CACHE_DIR"]

    parser = argparse.ArgumentParser(description="main")
    parser.add_argument(
        "--cluster-config",
        required=True,
        help="path to YAML cluster config (e.g. configs/della.yaml)",
    )
    parser.add_argument(
        "--work_dir",
        required=True,
        type=str,
        help="work directory where all experiments live",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="experiment name to create output directory in work dir",
    )
    parser.add_argument(
        "--aug-type",
        type=str,
        default="none",
        help="augmentation type to use",
        choices=[
            "none",
            "ours",
            # SA over the differentiable ops + the gradient cross-correlation
            # pipeline. See --no-corr-sa for the control arm.
            "grad_corr",
            "default",
            "random",
            "autoaugment",
            "augmix",
            "randaugment",
            "trivialaugment",
            "idbh",
            "vip",
        ],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="pspnet",
        help="backbone to use",
        choices=SUPPORTED_BACKBONES,
    )
    parser.add_argument(
        "--use-foundation-backbone",
        action="store_true",
        default=False,
        help="use foundation model backbone (dinov2)",
    )
    parser.add_argument(
        "--geometric-only",
        action="store_true",
        default=False,
        help="use geometric transforms only",
    )
    parser.add_argument(
        "--photometric-only",
        action="store_true",
        default=False,
        help="use photometric transforms only",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help="no clean training warmup",
    )
    parser.add_argument(
        "--random-aug",
        action="store_true",
        default=False,
        help="random augmentation with our method",
    )
    parser.add_argument(
        "--weighted-augs",
        action="store_true",
        default=False,
        help="augs are not treated equally",
    )
    parser.add_argument(
        "--freeze-early-layers",
        action="store_true",
        default=False,
        help="whether or not to freeze early layers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        help="dataset to train on",
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--sa_interval",
        type=int,
        default=None,
        help="interval of iterations to re-compute sa",
    )
    parser.add_argument(
        "--round_interval",
        type=int,
        default=None,
        help="interval of iterations to re-evaluate perturbation robustness (the SA "
        "pipeline's clock). Overrides schedule.round_interval in the cluster config. "
        "Defaults to max_iters // 20.",
    )
    parser.add_argument(
        "--descending-MA",
        action="store_true",
        default=False,
        help="whether to prioritize less severe augmentations",
    )
    parser.add_argument(
        "--uniform", action="store_true", default=False, help="use uniform augmentation"
    )
    parser.add_argument(
        "--no-corr-sa",
        action="store_true",
        default=False,
        help="under --aug-type=grad_corr, disable the sensitivity-analysis loop. "
        "Training then runs unaugmented and the gradient probe uses the fixed "
        "reference magnitude (0.5) for every op instead of the SA-derived "
        "distribution. This is the control arm for the correlation matrix.",
    )
    parser.add_argument(
        "--corr-magnitude-mode",
        type=str,
        default="mode",
        choices=["mode", "sampled_shared", "sampled_independent", "fixed"],
        help="how each op's probe magnitude is drawn from its SA distribution. "
        "'mode' (default) uses the modal level, constant across the batch. "
        "'sampled_shared' draws per image with one shared quantile across ops. "
        "'sampled_independent' lets each op draw on its own (attenuates R). "
        "'fixed' always uses the reference magnitude. Ignored without a snapshot.",
    )
    parser.add_argument(
        "--corr-magnitudes",
        type=str,
        default=None,
        help="path to a corr_magnitudes.json written by an earlier run. Its last "
        "snapshot seeds the probe magnitudes, so an --no-corr-sa control arm can be "
        "measured at the SAME magnitudes as the SA-on run it is compared against. "
        "Without it the control probes at the fixed 0.5 and the two matrices are "
        "not directly comparable. A live SA snapshot supersedes it.",
    )
    parser.add_argument(
        "--corr-sync-sa",
        action="store_true",
        default=False,
        help="fire the gradient sweep on the SA loop's clock (--round_interval) "
        "instead of its own --corr-interval. The sweep still runs from "
        "after_train_iter, which is BEFORE that round's val loop updates the pdf, "
        "so it probes at the previous round's magnitudes.",
    )
    parser.add_argument(
        "--corr-interval",
        type=int,
        default=None,
        help="interval of iterations between gradient sweeps and cross-correlation "
        "matrix emissions. Overrides schedule.corr_interval in the cluster config. "
        "Independent of --round_interval. Defaults to max_iters // 4. Ignored when "
        "--corr-sync-sa is set.",
    )
    parser.add_argument(
        "--corr-lambda",
        type=float,
        default=0.0,
        help="strength of the redundancy down-weighting applied to the training "
        "pdf: q(a) proportional to pdf(a) * exp(-lambda * red(a)), where red(a) is "
        "the standardized row sum of the correlation matrix R. 0 (the default) "
        "leaves the pdf bit-identical and is the control arm. Because red(a) is "
        "standardized, lambda means the same thing across runs and checkpoints: on "
        "the logged matrices 0.25 gives a max/min spread of 2.3-3.1x, 0.5 gives "
        "5-10x, and 1.0 is already extreme. Needs an R-keyed vocabulary, i.e. "
        "--aug-type=grad_corr.",
    )
    parser.add_argument(
        "--corr-red-mode",
        type=str,
        default="squared",
        choices=["squared", "abs", "signed"],
        help="how a row of R reduces to one redundancy score per op. 'squared' "
        "(default) is closure-proof and independent of any op's sign convention. "
        "'abs' and 'signed' are ablation arms -- note 'signed' PROTECTS an "
        "anti-correlated pair rather than down-weighting it.",
    )
    parser.add_argument(
        "--corr-lambda-ramp",
        type=str,
        default="linear",
        choices=["linear", "constant"],
        help="whether lambda ramps from 0 to --corr-lambda over training (default) "
        "or applies at full strength from the first emission. R measured early "
        "describes a model that barely discriminates between augmentations yet, so "
        "the ramp acts least on the least trustworthy measurement.",
    )
    parser.add_argument(
        "--corr-keep-within-op",
        action="store_true",
        default=False,
        help="keep the lighter/darker and _pos/_neg cells in red(a). They are "
        "excluded by default: the two directions of one op measure a "
        "parameterization convention, not redundancy between augmentations anyone "
        "would have chosen independently.",
    )
    parser.add_argument(
        "--adamw",
        action="store_true",
        default=False,
        help="whether to use AdamW optimizer",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument(
        "--no-inv-aug",
        action="store_true",
        default=False,
        help="exclude color augmentations",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--auto-scale-lr",
        action="store_true",
        help="Whether to scale the learning rate automatically. It requires "
        "`auto_scale_lr` in config, and `base_batch_size` in `auto_scale_lr`",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = f"ours_{args.backbone}_{args.dataset}"
        # args.exp_name = f"none_{args.backbone}_{args.dataset}" if args.aug_type is None \
        #                                                     else f"{args.aug_type}_{args.backbone}_{args.dataset}"

    # Set up working dir to save files and logs.
    if "ours" not in args.exp_name and args.aug_type == "ours":
        args.exp_name = args.exp_name + "_ours"

    # Same treatment for grad_corr, so the SA-on and SA-off arms can never land in
    # the same work_dir -- they write the same log files, and a silent collision
    # would interleave two incomparable sets of R matrices in corr_matrix_log.json.
    if args.aug_type == "grad_corr":
        suffix = "gradcorr_nosa" if args.no_corr_sa else "gradcorr"
        if suffix not in args.exp_name:
            args.exp_name = args.exp_name + "_" + suffix

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    torch.cuda.device(args.local_rank)

    train(args)

    dist_print("Done.")
