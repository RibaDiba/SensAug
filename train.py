import os
import numpy as np
import re
import glob
import json
import logging
import shutil
import zlib

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
from sensaug.dataset.gpu_augment import (  # noqa:F401
    GpuAugSegDataPreProcessor,
    set_train_spec,
)
from sensaug.dataset.idbh import IDBHTransform  # noqa:F401
from sensaug.dataset.vip import VIPAugTransform  # noqa:F401
from sensaug.hooks import *  # noqa:F403
from sensaug.loops import *  # noqa:F403
# Explicit for the same reason DOWNWEIGHT_METHODS below is: these are READ here,
# to turn configs/rounds.yaml into the correlation pipeline's firing schedule.
from sensaug.round_schedule import (
    DEFAULT_ROUNDS_CONFIG,
    load_round_config,
    resolve_schedule,
)
# Explicit rather than left to the star import above: this one is READ here, to
# build the flag's `choices` from the registry so the two cannot drift. A name
# that exists only by virtue of a star import is invisible to every linter that
# would otherwise catch it going stale.
from sensaug.loops.grad_corr_loop import DOWNWEIGHT_METHODS, HARD_PRUNING_METHODS
from sensaug.visualizer import BPSegLocalVisualizer  # noqa:F401

#: The arms that draw from the 32-op bank. All of them run on the GPU set, so
#: they differ only in how they pick and weight from it -- which is the whole
#: point of comparing them. "random" is deliberately not here: it is not a
#: compared arm and stays on the historical 20-op CPU vocabulary.
GPU_AUG_TYPES = ("ours", "default", "grad_corr")


def uses_gpu_augmentation(aug_type: str) -> bool:
    return aug_type in GPU_AUG_TYPES


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


def resolve_pruned_augmentations(cli_value):
    """Resolve the pruned-op list: CLI flag > cluster config's `pruned_augmentations:`.

    `cli_value` is `None` when `--pruned-augmentations` was never passed, which
    means "use the cluster config's list as-is". An explicit CLI value --
    including an explicit empty one, e.g. `--pruned-augmentations` alone --
    overrides the config list rather than merging with it, matching
    `resolve_interval`'s CLI > config precedence for the schedule.
    """
    if cli_value is not None:
        return list(cli_value)
    return list(PRUNED_AUGMENTATIONS)


def validate_pruned_augmentations(names, known):
    """Raise ValueError if any of `names` isn't a real augmentation op.

    `known` is the full vocabulary of op names that exist ANYWHERE in the
    codebase (union across all three perturbation sets), not just the ones
    active for the current --aug-type -- this is a typo check, not an
    applicability check. Pulled out of the argparse block so it's testable on
    its own; the CLI call site turns this into `parser.error(...)` so a bad
    name is caught before any config is built or work_dir created.
    """
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f"--pruned-augmentations names not found in any perturbation set: "
            f"{unknown}. Valid names: {sorted(known)}"
        )


def warn_ignored_pruned_augmentations(args, active_vocab):
    """Warn (never raise) about a pruned name that exists but is inert here.

    A name can be valid (it's in `validate_pruned_augmentations`'s union of
    every known op) yet belong to a perturbation set this run never touches --
    e.g. a snake_case diff32/non-diff32 name pruned on --aug-type=random,
    which only ever samples from legacy20's PascalCase names. That is not an
    error, but a silent no-op here would be a confusing one to debug later,
    so it gets one line in the run's own log (same posture and call site as
    warn_ignored_downweight_method -- after Runner.from_cfg, so it lands in
    {work_dir}/<timestamp>/<timestamp>.log rather than only the SLURM .out).
    """
    inert = [n for n in args.pruned_augmentations if n not in active_vocab]
    if not inert:
        return

    print_log(
        f"[pruned-augmentations] {inert} are valid op names but not part of "
        f"the perturbation set this run (--aug-type={args.aug_type}) actually "
        f"samples from, so pruning them has no effect here.",
        logger="current",
        level=logging.WARNING,
    )


def _active_pruning_vocab(args):
    """The op-name vocabulary this run actually samples from, for the pruned-
    augmentations inertness warning.

    Mirrors the perturbation_set choices build_config makes when it assembles
    the training pipeline (the `augmentation_type in (...)` branch around the
    "random"/"ours"/"grad_corr"/"default" pipeline entries, and
    cfg.val_cfg.perturbation_set for the SA-loop arms): "random" samples from
    the 20 PascalCase LEGACY20_OPS names; "ours"/"grad_corr"/"default" all
    train on the 32 shared diff32/non-diff32 snake_case names (DIFF32_OPS).
    Every other --aug-type (none, autoaugment, augmix, randaugment,
    trivialaugment, idbh, vip) does not sample from either perturbation-set
    registry at all, so pruning is unconditionally inert there -- returning an
    empty set flags every pruned name as such.
    """
    if args.aug_type == "random":
        return set(LEGACY20_OPS)
    if args.aug_type in ("ours", "grad_corr", "default"):
        return set(DIFF32_OPS)
    return set()


#: The random-pruning arms. `none` is the default -- no random prune at all, so
#: every invocation that predates this flag lands there unchanged. `null` is the
#: control arm for mRMR: drop N ops chosen at random, once, and keep them dropped
#: for the whole run. A tuple rather than a dispatch dict like DOWNWEIGHT_METHODS
#: because there is no per-arm function to dispatch TO -- the arm differs only in
#: how the op names are chosen, after which it is an ordinary
#: --pruned-augmentations run and every existing consumer handles it unchanged.
RANDOM_PRUNE_METHODS = ("none", "null")

#: The color/photometric ops --no-inv-aug removes, per vocabulary. Mirrors
#: RobustValLoop._remove_H_names (sensaug/loops/sensaug_loop.py) -- restated here
#: rather than imported because the loop resolves them from an instance that does
#: not exist yet at argparse time, and the two have to agree for a random draw's
#: count to mean what it says.
_REMOVE_H_NAMES = {
    "diff32": ("lighter_H", "darker_H"),
    "legacy20": ("PosterizeTransform", "SolarizeTransform"),
}


def _random_prune_set_name(aug_type):
    """Which perturbation registry a random prune draws from, or None.

    The same mapping `_active_pruning_vocab` makes, expressed as the
    `resolve_perturbation_set` key rather than the resolved name set, because the
    pool also has to respect --geometric-only / --photometric-only and those
    filters live inside that function.
    """
    if aug_type == "random":
        return "legacy20"
    if aug_type in ("ours", "grad_corr", "default"):
        return "diff32"
    return None


def random_prune_pool(args, already_pruned=()):
    """The ops a `null` draw is allowed to remove, sorted.

    Deliberately narrower than "every op in the vocabulary", so that dropping N
    means the run really does train on N fewer ops than its control:

    * --geometric-only / --photometric-only already restrict what is sampled,
      and resolve_perturbation_set applies them.
    * --no-inv-aug removes two ops at the SA curve's source (update_sa_curve's
      `exclude=`), and job_scripts/train_nexus_gamma.sbatch passes it on EVERY
      run. Drawing one of those would spend a prune on an op that was already
      gone, leaving this arm N-1 real prunes against an mRMR arm's N.
    * anything --pruned-augmentations or the cluster config already removed, so a
      random draw COMPOSES with an explicit list instead of overlapping it and
      quietly shrinking its own count.

    Empty for an --aug-type that samples from no perturbation registry at all
    (none, autoaugment, ...), which is what lets one bounds check on `count` also
    reject those arms.
    """
    set_name = _random_prune_set_name(args.aug_type)
    if set_name is None:
        return []

    excluded = set(already_pruned)
    if args.no_inv_aug:
        excluded.update(_REMOVE_H_NAMES[set_name])

    names = resolve_perturbation_set(
        set_name,
        geometric_only=args.geometric_only,
        photometric_only=args.photometric_only,
    )
    return sorted(n for n in names if n not in excluded)


def _seed_from_text(text):
    """A stable 32-bit seed from an arbitrary run id.

    zlib.crc32 rather than hash(): PYTHONHASHSEED is randomized per process, so
    hash() of the same SLURM job id would differ between ranks -- which is the
    exact failure resolve_random_prune_seed exists to prevent.
    """
    try:
        return int(text) % (2**32)
    except ValueError:
        return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def resolve_random_prune_seed(cli_value, env=None):
    """Resolve the `null` arm's seed to `(seed, source)`. Raises ValueError.

    Every rank runs train.py independently under torchrun, and at argparse time
    torch.distributed is NOT yet initialized -- that happens inside
    Runner.from_cfg, long after this. So there is no collective here to sync a
    seed with, and mmengine.dist.sync_random_seed() does not help: with dist
    uninitialized it sees world_size == 1 and returns a per-process seed without
    broadcasting anything. A freshly generated seed would therefore give every
    rank a DIFFERENT pruned set -- four banks averaged into one gradient update,
    with nothing in the logs saying so.

    Hence: the seed is either stated outright, or derived from something every
    rank of the same job already agrees on and that still differs between runs.
    If neither is available and this is demonstrably multi-rank, refuse rather
    than guess.
    """
    env = os.environ if env is None else env

    if cli_value is not None:
        return int(cli_value) % (2**32), "--random-prune-seed"

    for key in ("SLURM_JOB_ID", "TORCHELASTIC_RUN_ID"):
        raw = env.get(key)
        if raw:
            return _seed_from_text(raw), key

    world_size = env.get("WORLD_SIZE")
    if world_size in (None, "", "1"):
        return int.from_bytes(os.urandom(4), "little"), "urandom"

    raise ValueError(
        f"--random-prune-method=null needs a seed that every rank agrees on, and "
        f"this launch has WORLD_SIZE={world_size} with neither SLURM_JOB_ID nor "
        f"TORCHELASTIC_RUN_ID set to derive one from. Generating one here would "
        f"give each rank a different pruned set, which no log would reveal. Pass "
        f"--random-prune-seed explicitly."
    )


def draw_random_prune(pool, count, seed):
    """Draw `count` op names from `pool`, deterministically for a given seed.

    A private RandomState rather than the global numpy RNG, which set_manual_seed
    pins to 0 for the run proper and which GpuAugSegDataPreProcessor then draws
    every per-image augmentation from -- consuming from it here would shift every
    subsequent draw as a function of how many ops were pruned, so the two arms
    would differ by more than their banks. `pool` is sorted, so the result cannot
    depend on dict iteration order either.
    """
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(list(pool), size=count, replace=False).tolist())


def reject_random_prune(args, pool=None):
    """Return a rejection message for the random-pruning flags, or None.

    Returns a string rather than raising so the CLI can route it through
    `parser.error` -- exit 2, before any config is built or work_dir created --
    while the tests can assert on it directly. Same posture as
    `reject_skip_pruned_eval`.

    Called twice: once without `pool` for the checks that need no vocabulary, and
    again with it for the bounds check, so a bad flag combination is rejected
    before the pool is even resolved.
    """
    method = args.random_prune_method
    count = args.random_prune_count

    if method == "none":
        stray = [
            name
            for name, value in (
                ("--random-prune-count", count),
                ("--random-prune-seed", args.random_prune_seed),
            )
            if value is not None
        ]
        if stray:
            return (
                f"{', '.join(stray)} set without --random-prune-method=null, so "
                f"nothing would be pruned at random. An arm is never left "
                f"unnamed -- the same rule --corr-downweight-method applies on "
                f"grad_corr, and for the same reason: the arm is not recoverable "
                f"from the checkpoint afterwards."
            )
        return None

    if count is None:
        return (
            "--random-prune-method=null requires --random-prune-count N, the "
            "number of ops to drop at random. It has no default: the point of "
            "the arm is to match some specific mRMR run's prune count, and "
            "guessing one would make the comparison meaningless."
        )

    if count < 1:
        return f"--random-prune-count must be >= 1, got {count}."

    if pool is None:
        return None

    if not pool:
        return (
            f"--random-prune-method=null has nothing to draw from on "
            f"--aug-type={args.aug_type}: it samples from no perturbation set, "
            f"so there are no ops to prune. Use ours, grad_corr, default or "
            f"random."
        )

    if count >= len(pool):
        # >= rather than >: pruning the whole pool leaves a pdf of nothing but
        # ("none", 0), and CollectGradientHook refuses a static prune naming
        # every op outright. Failing here says why; failing there says less.
        flags = "".join(
            f", {flag}"
            for flag, on in (
                ("--no-inv-aug", args.no_inv_aug),
                ("--geometric-only", args.geometric_only),
                ("--photometric-only", args.photometric_only),
            )
            if on
        )
        return (
            f"--random-prune-count={count} would leave no augmentations: only "
            f"{len(pool)} ops are eligible on this run (--aug-type="
            f"{args.aug_type}{flags}; already pruned: "
            f"{sorted(set(args.pruned_augmentations))}). Eligible ops: {pool}"
        )

    return None


def random_prune_record(args):
    """The reproducibility record for a `null` run, as a JSON-able dict.

    Neither the seed nor the draw is recoverable from a checkpoint, and the seed
    is not necessarily in the launch command either (it can be derived from the
    SLURM job id), so the arm has to write itself down or it cannot be repeated.
    """
    dropped = set(args.random_prune_ops)
    return {
        "method": args.random_prune_method,
        "count": args.random_prune_count,
        "seed": args.random_prune_seed_used,
        "seed_source": args.random_prune_seed_source,
        "aug_type": args.aug_type,
        "perturbation_set": _random_prune_set_name(args.aug_type),
        "pool_size": len(args.random_prune_pool),
        "pool": list(args.random_prune_pool),
        "dropped": list(args.random_prune_ops),
        "kept": [n for n in args.random_prune_pool if n not in dropped],
        "pruned_augmentations": list(args.pruned_augmentations),
    }


def log_random_prune(args, work_dir):
    """Record the `null` arm's draw, to the run's log and to its work_dir.

    Called from train() after Runner.from_cfg for two reasons: print_log(
    logger="current") only reaches {work_dir}/<timestamp>/<timestamp>.log once
    the runner's logger exists (same reason as warn_ignored_downweight_method),
    and is_main_process() only tells the truth once dist is initialized -- before
    that every rank believes it is rank 0 and all four would race on the same
    file.
    """
    if args.random_prune_method != "null":
        return

    record = random_prune_record(args)
    print_log(
        f"[random-prune] arm=null seed={record['seed']} "
        f"(from {record['seed_source']}), dropped {record['count']} of "
        f"{record['pool_size']} eligible ops: {record['dropped']}. "
        f"Training on: {record['kept']}",
        logger="current",
    )

    if is_main_process():
        with open(os.path.join(work_dir, "random_prune.json"), "w") as f:
            json.dump(record, f, indent=2)


def warn_ignored_downweight_method(args):
    """Warn when `--corr-downweight-method` was supplied but nothing will read it.

    The method is consumed by `GradCorrValLoop` and by nothing else, so on any arm
    that does not build one the flag is inert. Silently inert is the failure this
    warns about: the launch command reads as if a redundancy arm were configured,
    the run trains as the baseline, and the difference surfaces only as a result
    that inexplicably matches the control.

    Never raises. A flag that has no effect is not a reason to refuse to train --
    the run is still a perfectly good run of whatever `--aug-type` actually says.

    Called from `train()` AFTER Runner.from_cfg, not from `build_config`, and that
    ordering is the point: `print_log(logger="current")` only reaches the run's
    `{work_dir}/<timestamp>/<timestamp>.log` once the runner's logger exists.
    Warning earlier would put it in stdout alone, i.e. in the SLURM .out file that
    nobody reads six weeks later while trying to work out what an experiment was.
    """
    if args.corr_downweight_method is None:
        return

    if args.aug_type != "grad_corr":
        why = (
            f"--aug-type={args.aug_type} builds no correlation pipeline, so nothing "
            f"publishes a redundancy score and there is no GradCorrValLoop to apply "
            f"one. Use --aug-type=grad_corr to enable it."
        )
    elif args.no_corr_sa:
        # Mandatory on every grad_corr run, including this one -- so unlike the
        # branch above, this fires on a correctly-formed command. It is still worth
        # saying: --no-corr-sa is the control arm, it builds mmengine's stock
        # ValLoop, and R is measured there but never fed back into training.
        why = (
            "--no-corr-sa disables the sensitivity-analysis loop, so the stock "
            "ValLoop is built instead of GradCorrValLoop. The correlation matrix is "
            "still measured; it just does not reweight anything on this arm."
        )
    else:
        return

    print_log(
        f"[downweight] --corr-downweight-method="
        f"{args.corr_downweight_method} is set but will NOT be applied: {why}",
        logger="current",
        level=logging.WARNING,
    )


SKIP_PRUNED_EVAL_REJECTION = (
    "--corr-skip-pruned-eval is not supported on --aug-type=grad_corr.\n\n"
    "It would skip the gradient sweep for whatever ops the down-weighting "
    "method currently has pruned and backfill each of them from "
    "CollectGradientHook._last_full_row -- gradients measured at an EARLIER "
    "checkpoint. Those rows are then correlated against freshly measured ones, "
    "so a single R mixes vintages and successive emissions are no longer "
    "comparable to each other. Watching R change as the model improves is what "
    "a grad_corr session is for, so this is not a saving the pipeline can take.\n\n"
    "If the goal is to stop training on an op: prune it statically with "
    "--pruned-augmentations. That excludes it from the SA round-eval, the "
    "training pdf and the gradient sweep alike, and its row of R is recorded as "
    "'static_pruned' (never measured) rather than backfilled."
)


def reject_skip_pruned_eval(args):
    """Return a rejection message for `--corr-skip-pruned-eval`, or None.

    Errors on `grad_corr` -- the ONE arm where the flag could act -- for the
    reason spelled out in SKIP_PRUNED_EVAL_REJECTION: skipping trades a
    comparable R for compute the measurement cannot spare.

    Returns a string rather than raising so the CLI can route it through
    `parser.error` (exit 2, before any config is built or work_dir created)
    while the tests can assert on it directly.
    """
    if not args.corr_skip_pruned_eval or args.aug_type != "grad_corr":
        return None
    return SKIP_PRUNED_EVAL_REJECTION


def warn_ignored_skip_pruned_eval(args):
    """Warn when `--corr-skip-pruned-eval` was supplied on an arm that ignores it.

    `runner.corr_pruned_ops` is only ever non-empty under a
    HARD_PRUNING_METHODS arm (today, just mRMR) running `GradCorrValLoop`, so on
    every other arm this flag is a no-op skip-list that's always empty: correct,
    but silently so, which reads exactly like the optimization firing when it
    never has anything to skip. `grad_corr` itself never reaches here --
    `reject_skip_pruned_eval` has already stopped the run at the CLI.

    Same posture as warn_ignored_downweight_method: never raises, called from
    train() after Runner.from_cfg so the warning lands in the run's own log
    file rather than only the SLURM .out.
    """
    if not args.corr_skip_pruned_eval:
        return

    print_log(
        f"[skip-pruned-eval] --corr-skip-pruned-eval is set but will NOT skip "
        f"anything: --aug-type={args.aug_type} builds no GradCorrValLoop, so "
        f"nothing ever publishes runner.corr_pruned_ops for either sweep to "
        f"read. To exclude an op from this run, use --pruned-augmentations.",
        logger="current",
        level=logging.WARNING,
    )


def warn_ignored_hold_none_prob(args):
    """Warn when `--hold-none-prob` was supplied but there is no drift to hold.

    The flag only does anything when --pruned-augmentations removed at least one
    op from the vocabulary this run samples from: with nothing pruned the
    denominator it pins is already the surviving count. Silent inertness here
    would be the bad kind -- the launch command reads as if the augmentation
    rate had been controlled for.
    """
    if not args.hold_none_prob:
        return
    effective = set(args.pruned_augmentations) & _active_pruning_vocab(args)
    if effective:
        return

    print_log(
        f"[hold-none-prob] --hold-none-prob is set but no op is pruned from the "
        f"vocabulary --aug-type={args.aug_type} samples from, so P(none) is "
        f"already at its unpruned value and nothing is held.",
        logger="current",
        level=logging.WARNING,
    )


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
            # grad_corr/ours/default all train on "diff32": the differentiable ops
            # themselves, applied batched on GPU after collation. That makes the
            # function training applies identical to the one the gradient probe
            # differentiates, so a per-op score read off R indexes straight into
            # the training pdf and the magnitudes mean the same thing on both
            # sides. Keeping all three arms on one set is also what makes them
            # comparable -- they're meant to differ in how they pick and weight
            # from the bank, not in which ops exist or what a magnitude buys.
            #
            # "random" stays on "legacy20" on purpose: it isn't a compared arm.
            #
            # The GPU set has no pipeline transform at all -- see
            # sensaug/dataset/gpu_augment.py -- so nothing is appended for it here.
            # The training policy is installed on the data preprocessor instead,
            # by the SA loop via runner_utils' GPU dispatch.
            if augmentation_type == "random":
                pipeline.append(
                    dict(
                        type="RandomAlphaTrainTransform",
                        geometric_only=args.geometric_only,
                        photometric_only=args.photometric_only,
                        perturbation_set="legacy20",
                        pruned=tuple(args.pruned_augmentations),
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

    # The val-round grid, from configs/rounds.yaml. Cluster-independent, unlike
    # everything in --cluster-config: how many rounds a run has and where the
    # correlation pipeline fires on them is an experiment parameter, not a path.
    round_cfg = load_round_config(args.rounds_config)

    # round_interval drives the SA pipeline (RobustValLoop; its SA-curve recompute
    # is every SA_CURVE_CADENCE of these rounds, in sensaug/loops/sensaug_loop.py).
    # rounds.yaml's n_rounds is only the DEFAULT divisor -- the CLI flag and the
    # cluster config's schedule: block still win.
    round_interval = resolve_interval(
        args.round_interval,
        "round_interval",
        cfg.train_cfg.max_iters // round_cfg.n_rounds,
    )
    cfg.train_cfg.val_interval = round_interval

    # The correlation pipeline's clock. `None` -- neither --corr-interval nor
    # schedule.corr_interval given -- is the DEFAULT and means "use the SA round
    # grid" (the schedule built below), not "use max_iters // 4": an emission is
    # only worth taking where a pdf can read it. Naming an interval opts back out
    # into a clock that is independent of the rounds.
    corr_interval = resolve_interval(args.corr_interval, "corr_interval", None)

    # The rounds this run will REALLY have, which is round_cfg.n_rounds only when
    # nothing overrode the interval above. The schedule is built from this, not
    # from the configured number: derived from the wrong one, every round would
    # still look right in the log while pointing at the wrong iteration.
    n_rounds = cfg.train_cfg.max_iters // round_interval
    warmup_rounds = 0 if args.no_warmup else round_cfg.warmup_rounds

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
        # GradCorrValLoop is RobustValLoop plus Lever 3's redundancy
        # down-weighting -- the only thing it adds. It is a separate registry name
        # rather than a flag on RobustValLoop so that the SA arm cannot be handed
        # correlation-pipeline kwargs it has no consumer for.
        cfg.val_cfg.type = (
            "GradCorrValLoop" if args.aug_type == "grad_corr" else "RobustValLoop"
        )
        # "diff32": the SA round-eval probes exactly the ops training applies and
        # the probe differentiates. On the GPU set the round-eval sets preprocessor
        # state instead of rebuilding the val dataloader once per (op, level) pair,
        # which is what made this phase the dominant cost of a grad_corr run.
        cfg.val_cfg.perturbation_set = (
            "diff32" if args.aug_type in ("grad_corr", "ours") else "legacy20"
        )
        cfg.val_cfg.ratio = eval_ratio
        cfg.val_cfg.sa_curve_path = "sensaug/testing/test_levels_voc.json"
        cfg.val_cfg.uniform = args.uniform
        # cfg.val_cfg.uniform = False
        cfg.val_cfg.descending_MA = args.descending_MA  # defaults to False
        # cfg.val_cfg.descending_MA = False # NOTE: False --> severe augmentations prioritized in pdf
        cfg.val_cfg.remove_H = args.no_inv_aug
        cfg.val_cfg.warmup_rounds = warmup_rounds
        cfg.val_cfg.random_aug = args.random_aug
        cfg.val_cfg.geometric_only = args.geometric_only
        cfg.val_cfg.photometric_only = args.photometric_only
        cfg.val_cfg.weighted_augs = args.weighted_augs
        cfg.val_cfg.pruned_augmentations = args.pruned_augmentations
        cfg.val_cfg.hold_none_prob = args.hold_none_prob
        # Lever 3's two knobs only exist on GradCorrValLoop. Setting them
        # unconditionally would attach kwargs RobustValLoop does not accept, and
        # under --no-corr-sa (sa_loop False) there is no custom val loop at all --
        # mmengine builds the stock ValLoop and would raise on them.
        if args.aug_type == "grad_corr":
            cfg.val_cfg.corr_lambda = args.corr_lambda
            cfg.val_cfg.corr_lambda_ramp = args.corr_lambda_ramp
            # No `or "exp"` fallback: the CLI gate below makes this non-None on
            # every grad_corr run, and GradCorrValLoop rejects None outright. An
            # implicit default here would be a silent arm -- unrecoverable from the
            # checkpoint, the logs or the work_dir name after the fact.
            cfg.val_cfg.corr_downweight_method = args.corr_downweight_method
            # Always False: reject_skip_pruned_eval has already stopped the run
            # at the CLI if it was asked for. Passed explicitly rather than
            # dropped so the loop's signature stays the shape the tests exercise.
            cfg.val_cfg.corr_skip_pruned_eval = False
        # cfg.val_cfg.remove_H = False
        cfg.test_cfg.type = "SubsetTestLoop"
        cfg.test_cfg.ratio = eval_ratio
        cfg.train_cfg.type = "RobustIterBasedTrainLoop"
        cfg.train_cfg.init_sa = True if (cfg.resume or args.no_warmup) else False

        # NOTE: LoveDA doesn't converge very well on default lr; we should reduce it by 1 order mag
        if "loveda" in args.dataset.lower():
            cfg.optimizer.lr *= 0.1

    if args.aug_type == "grad_corr":
        # Gradient-based augmentation cross-correlation. At each firing iteration
        # CollectGradientHook freezes the model and sweeps the whole clean val set
        # for d loss / d magnitude, then
        # PerturbationSensitivityAnalysisHookWithGradients correlates that sweep
        # into R. Both are handed the SAME gate, built once here rather than twice.
        #
        # ORDERING, and every mode below depends on it: IterBasedTrainLoop calls
        # val_loop.run() AFTER run_iter, so a sweep landing on a val iteration fires
        # BEFORE that round's val loop rebuilds the pdf. It therefore probes at the
        # PREVIOUS round's magnitudes (the ones in effect over the window being
        # measured, which is the right semantics) and its R is on the runner in time
        # for THIS round's pdf to be reweighted by it.
        #
        # Three ways to say when, in precedence order:
        emit_interval = None
        fire_iters = control_iters = full_recheck_iters = ()
        if args.corr_sync_sa:
            # Every SA round. The densest schedule available -- one R per pdf
            # rebuild, at ~n_rounds/4 times the cost of the default.
            emit_interval = round_interval
        elif corr_interval is not None:
            # An explicitly named interval: a clock independent of the rounds, which
            # is what this pipeline had by default before the round-aligned
            # schedule. Emissions land wherever the arithmetic puts them, so an R
            # can be measured mid-curve and sit unread until the next round.
            emit_interval = corr_interval
        else:
            # THE DEFAULT: the round grid from configs/rounds.yaml. One baseline
            # probe in the last warmup round, then one emission per SA-curve
            # recompute, each governing the rounds that curve governs. See
            # sensaug/round_schedule.py for why those rounds and not others.
            #
            # `pre_train_round` mirrors RobustIterBasedTrainLoop's `init_sa`: those
            # runs spend round 0 on a val pass before training starts, which shifts
            # every subsequent round one interval earlier.
            pre_train_round = sa_loop and bool(cfg.resume or args.no_warmup)
            schedule = resolve_schedule(
                round_cfg,
                n_rounds=n_rounds,
                warmup_rounds=warmup_rounds,
                round_interval=round_interval,
                pre_train_round=pre_train_round,
            )
            fire_iters = schedule.fire_iters
            control_iters = schedule.control_iters
            full_recheck_iters = schedule.full_recheck_iters
            for warning in schedule.warnings:
                dist_print(f"WARNING: {warning}")
            # In the launch log so an experiment records the schedule it actually
            # ran, not the one rounds.yaml happened to say at analysis time.
            dist_print(
                f"Correlation schedule ({'derived' if schedule.derived else 'explicit'}"
                f", {round_cfg.path}): rounds {list(schedule.rounds)} of {n_rounds} "
                f"-> iters {list(fire_iters)} (control: {list(control_iters)})"
            )

        # (The two --corr-skip-pruned-eval schedule warnings that used to sit here
        # are gone with the flag: reject_skip_pruned_eval stops that combination
        # at the CLI, so neither could ever fire.)

        # The priorities are load-bearing, not cosmetic: both hooks act in
        # after_train_iter, and the correlation hook must see the sweep the
        # collector just wrote. NORMAL (50) runs before LOW (70).
        cfg.custom_hooks = (cfg.get("custom_hooks") or []) + [
            dict(
                type="CollectGradientHook",
                interval=emit_interval,
                fire_iters=fire_iters or None,
                sweep_batch_size=1,
                magnitude_mode=args.corr_magnitude_mode,
                magnitudes_path=args.corr_magnitudes,
                skip_pruned=False,  # see reject_skip_pruned_eval
                full_recheck_iters=full_recheck_iters,
                # Statically pruned ops are excluded from the sweep entirely and
                # get an all-NaN row -- a separate mechanism from skip_pruned,
                # which backfills from cache. See the hook's class docstring.
                static_pruned_ops=args.pruned_augmentations,
                priority="NORMAL",
            ),
            dict(
                type="PerturbationSensitivityAnalysisHookWithGradients",
                interval=emit_interval,
                fire_iters=fire_iters or None,
                control_iters=control_iters,
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

    # The GPU set has no pipeline transform: its ops are applied batched, after
    # collation, by a SegDataPreProcessor subclass. Swapping the type here covers
    # every backbone at once -- each one declares its own data_preprocessor in
    # sensaug/custom_configs/mmseg/_base_/models/, and they differ only in
    # mean/std, which is carried through untouched.
    if uses_gpu_augmentation(args.aug_type):
        cfg.model.data_preprocessor.type = "GpuAugSegDataPreProcessor"
        cfg.model.data_preprocessor.pruned_ops = args.pruned_augmentations

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
    # Same reason as the line above: the run should carry the schedule it was
    # launched with, so a later analysis does not have to trust that
    # configs/rounds.yaml still says what it said months ago.
    shutil.copy(args.rounds_config, os.path.join(cfg.work_dir, "rounds.yaml"))
    runner = Runner.from_cfg(cfg)
    set_manual_seed(0)  # set seed
    runner.val_loop  # initialize val loop
    runner.test_loop

    # After the runner exists so it lands in the run's own log file, and after
    # val_loop so a bad method name has already failed hard rather than being
    # reported as merely ignored.
    warn_ignored_downweight_method(args)
    warn_ignored_skip_pruned_eval(args)
    warn_ignored_pruned_augmentations(args, _active_pruning_vocab(args))
    warn_ignored_hold_none_prob(args)
    log_random_prune(args, cfg.work_dir)

    # Install the initial uniform training policy for the GPU arms.
    #
    # This is what RandomAlphaTrainTransform used to do by sitting in the pipeline
    # from iteration 0: training augments uniformly over the bank until the SA loop
    # has enough rounds to publish a pdf, at which point RobustValLoop replaces
    # this policy with that pdf. Doing it here rather than in the loop matters for
    # two reasons -- `default` builds no SA loop at all and would otherwise train
    # completely unaugmented, and `ours`/`grad_corr` would train clean through
    # their warmup rounds instead of uniformly augmented.
    if uses_gpu_augmentation(args.aug_type):
        set_train_spec(
            runner,
            geometric_only=args.geometric_only,
            photometric_only=args.photometric_only,
        )

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
    PRUNED_AUGMENTATIONS = _seg["PRUNED_AUGMENTATIONS"]

    parser = argparse.ArgumentParser(description="main")
    parser.add_argument(
        "--cluster-config",
        required=True,
        help="path to YAML cluster config (e.g. configs/della.yaml)",
    )
    parser.add_argument(
        "--rounds-config",
        type=str,
        default=DEFAULT_ROUNDS_CONFIG,
        help="path to the YAML val-round grid (default: configs/rounds.yaml). "
        "Sets how many val/SA rounds a run has, how many of them are warmup, and "
        "which of them the gradient cross-correlation pipeline fires on. Kept out "
        "of the cluster config on purpose -- it is an experiment parameter, and "
        "della.yaml and nexus.yaml should not each carry a copy that can drift.",
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
        help="fire the gradient sweep on EVERY SA round (--round_interval) rather "
        "than on the round-aligned subset the default schedule uses. ~20x the "
        "sweeps, one per pdf rebuild. The sweep still runs from after_train_iter, "
        "which is BEFORE that round's val loop updates the pdf, so it probes at the "
        "previous round's magnitudes. Takes precedence over --corr-interval.",
    )
    parser.add_argument(
        "--corr-interval",
        type=int,
        default=None,
        help="put the correlation pipeline on a fixed iteration clock of its own "
        "instead of the default SA-round-aligned schedule. Overrides "
        "schedule.corr_interval in the cluster config. By DEFAULT (neither given) "
        "sweeps fire on the SA rounds that can act on them: the last warmup round "
        "as an unpublished baseline probe, then every SA-curve recompute round -- "
        "rounds 3, 4, 10 and 16 of 20 at the default round_interval and warmup. "
        "Naming an interval here decouples the two again, so an R may be measured "
        "mid-curve and sit unread until the next round. Ignored when --corr-sync-sa "
        "is set.",
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
        "--corr-downweight-method",
        type=str,
        default=None,
        choices=sorted(DOWNWEIGHT_METHODS),
        help="which function turns the redundancy score into the reweighted "
        "training pdf. 'none' leaves the pdf exactly as the SA loop generated it "
        "-- R is still measured and logged, just not fed back. 'soft-weighting' "
        "is the max-entropy tilt q(a) ~ pdf(a)*exp(-lambda*red(a)), soft by "
        "construction: an op is pushed down but structurally cannot reach zero. "
        "'mRMR' HARD-PRUNES instead: it ranks the ops by minimum-Redundancy "
        "Maximum-Relevance (relevance = the SA loop's own pdf mass, redundancy = "
        "the pairwise cells of R) and sets every op outside the top "
        "ceil(A/(1+lambda)) to probability exactly zero -- lambda buys a smaller "
        "bank rather than a flatter one (0.25 keeps ~80% of the ops, 0.5 ~67%, "
        "1.0 half). Deletion is a stronger claim than the observed correlation "
        "sizes support, so read an mRMR run next to a soft-weighting run at the "
        "same lambda, and pair it with --photometric-only until the geometric "
        "ops' R stops being contaminated by image-label misalignment. "
        "This is the third axis of the correlation pipeline -- --corr-red-mode "
        "picks how a row of R reduces to one score per op, --corr-lambda picks "
        "how hard that score pushes, and this picks the form of the push. "
        "REQUIRED with --aug-type=grad_corr -- there is deliberately no default, "
        "so every correlation run names its arm in its own launch command. "
        "Ignored (with a warning) on every other --aug-type. New methods are "
        "added to DOWNWEIGHT_METHODS in sensaug/loops/grad_corr_loop.py.",
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
        "--corr-skip-pruned-eval",
        action="store_true",
        default=False,
        help="REJECTED on --aug-type=grad_corr, which is the only arm where it "
        "could ever act. It would skip the gradient sweep and the round-eval for "
        "whatever ops mRMR currently has pruned, backfilling each from its last "
        "real measurement -- but that makes a skipped op's row of R come from an "
        "OLDER checkpoint's gradients while every other row is fresh, so R mixes "
        "vintages and stops being comparable round over round. Watching R evolve "
        "as the model improves is the entire purpose of a grad_corr session, so "
        "the saving is not one this pipeline can take. Inert (warned, not "
        "rejected) on every other --aug-type: nothing but GradCorrValLoop ever "
        "publishes runner.corr_pruned_ops, so there is no skip-list to read. To "
        "actually train without an op, prune it statically with "
        "--pruned-augmentations, which excludes it everywhere and costs R "
        "nothing.",
    )
    parser.add_argument(
        "--hold-none-prob",
        action="store_true",
        default=False,
        help="hold P(no augmentation) fixed as --pruned-augmentations shrinks "
        "the bank. The training pdf carries a synthetic ('none', 0) entry whose "
        "mass is derived from the count of SURVIVING ops, so pruning 6 of 30 "
        "silently raises it by ~0.7pp -- the pruned arm then trains on clean "
        "images that much more often than its control, a second difference "
        "sitting inside the comparison. With this flag the denominator counts "
        "the pruned ops as if still present, so their mass goes to the surviving "
        "ops and the augmentation RATE is unchanged: the prune alters which "
        "augmentation is sampled, never how often one is. Same invariant "
        "--corr-downweight-method=mRMR already holds internally. Off by default, "
        "so a run launched before this flag existed is reproduced exactly. "
        "Inert (warned) without --pruned-augmentations. Applies to "
        "generate_pdf_new (the default) and --uniform; NOT to --weighted-augs, "
        "whose rate moves with the op count for a different reason (a truncated "
        "beta-binomial over all (op, level) pairs) that this flag does not "
        "address.",
    )
    parser.add_argument(
        "--pruned-augmentations",
        type=str,
        nargs="+",
        default=None,
        metavar="OP_NAME",
        help="op names to permanently exclude from sampling for this entire "
        "run, regardless of --aug-type. Overrides (does not merge with) the "
        "cluster config's `pruned_augmentations:` list. Valid names are "
        "LEGACY20_OPS' 20 PascalCase names or the 32 shared diff32/non-diff32 "
        "snake_case names in sensaug/dataset/augmentations.py -- an unknown "
        "name aborts before any config is built or work_dir created. A "
        "known name that isn't part of the vocabulary this run's --aug-type "
        "actually samples from is accepted but logged as inert.",
    )
    parser.add_argument(
        "--random-prune-method",
        type=str,
        default="none",
        choices=list(RANDOM_PRUNE_METHODS),
        help="the random-pruning arm. `none` (default) draws nothing and leaves "
        "the run exactly as it was. `null` is the control arm for mRMR: before "
        "training starts, pick --random-prune-count ops at random out of the "
        "ones this run would otherwise sample from, and drop them for the whole "
        "run -- so the comparison is 'does ranking ops by redundancy beat "
        "dropping the same number of them at random?'. Unlike mRMR the prune is "
        "fixed, never re-derived: it carries no R, no lambda and no correlation "
        "pipeline, and feeds straight into --pruned-augmentations.",
    )
    parser.add_argument(
        "--random-prune-count",
        type=int,
        default=None,
        metavar="N",
        help="how many ops --random-prune-method=null DROPS (not keeps). "
        "Required on that arm, no default. To match a finished mRMR stage-1 "
        "run, use the word count of its mrmr_pruned_ops.txt. Note the pool it "
        "draws from already excludes whatever --no-inv-aug, --geometric-only / "
        "--photometric-only and any explicit --pruned-augmentations removed, so "
        "N is always N ops fewer than the control trains on. As with mRMR, "
        "pruning raises P(no augmentation) unless --hold-none-prob is set: pass "
        "that flag on both arms of a comparison or on neither.",
    )
    parser.add_argument(
        "--random-prune-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="seed for --random-prune-method=null's draw. Optional: when "
        "omitted it is derived from SLURM_JOB_ID, else TORCHELASTIC_RUN_ID, "
        "else (single-process runs only) os.urandom. Whatever is used is logged "
        "and written to {work_dir}/random_prune.json. Pass it explicitly to "
        "repeat a draw, or to sweep several random draws of the same size "
        "(seeds 0, 1, 2 ...). It must be identical on every rank -- multi-rank "
        "launches with no derivable seed are refused rather than silently "
        "training a different bank per rank.",
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

    # A grad_corr run must name its down-weighting arm. Enforced at argparse time
    # rather than in build_config so it costs nothing: usage + exit 2, before any
    # config is built, any checkpoint is loaded or any work_dir is created.
    #
    # Applies to --no-corr-sa too, even though that arm builds no GradCorrValLoop
    # and will not read the value (warn_ignored_downweight_method says so at
    # startup). One rule for every grad_corr invocation is easier to hold than
    # "required except when", and it keeps every arm's command self-describing.
    if args.aug_type == "grad_corr" and args.corr_downweight_method is None:
        parser.error(
            "--aug-type=grad_corr requires --corr-downweight-method "
            f"(one of {sorted(DOWNWEIGHT_METHODS)}). It has no default: the "
            "down-weighting arm is not recoverable from the checkpoint or the "
            "logs afterwards, so it has to be stated up front."
        )

    # Resolve CLI > cluster config, then validate against every known op name
    # BEFORE any config is built, checkpoint loaded, or work_dir created --
    # same fail-fast posture as the grad_corr check above. Re-assigned onto
    # args so build_config can just read args.pruned_augmentations like every
    # other resolved flag.
    args.pruned_augmentations = resolve_pruned_augmentations(args.pruned_augmentations)
    try:
        validate_pruned_augmentations(
            args.pruned_augmentations, set(LEGACY20_OPS) | set(DIFF32_OPS)
        )
    except ValueError as e:
        parser.error(str(e))

    # Same fail-fast posture, and the same reason: rejecting a flag combination
    # after Runner.from_cfg would mean discovering it a compute node and a
    # scheduler queue later.
    rejection = reject_skip_pruned_eval(args)
    if rejection is not None:
        parser.error(rejection)

    # The `null` random-pruning arm. Resolved here, at argparse time, for three
    # reasons: it must land in args.pruned_augmentations before build_config
    # reads it (the SA curve's exclude=, the GPU sampling bank, the gradient
    # sweep's static_pruned_ops all come off that one list); it must be known
    # before the exp_name suffix below, which carries the seed; and a bad
    # combination should cost exit 2 rather than a compute node and a queue.
    args.random_prune_ops = []
    args.random_prune_pool = []
    args.random_prune_seed_used = None
    args.random_prune_seed_source = None

    rejection = reject_random_prune(args)
    if rejection is not None:
        parser.error(rejection)

    if args.random_prune_method == "null":
        args.random_prune_pool = random_prune_pool(args, args.pruned_augmentations)
        rejection = reject_random_prune(args, pool=args.random_prune_pool)
        if rejection is not None:
            parser.error(rejection)

        try:
            seed, seed_source = resolve_random_prune_seed(args.random_prune_seed)
        except ValueError as e:
            parser.error(str(e))

        args.random_prune_seed_used = seed
        args.random_prune_seed_source = seed_source
        args.random_prune_ops = draw_random_prune(
            args.random_prune_pool, args.random_prune_count, seed
        )
        # Union, not replacement: the draw already excluded everything on the
        # resolved list, so this composes the two rather than letting one hide
        # the other. Sorted so the list a run records is stable.
        args.pruned_augmentations = sorted(
            set(args.pruned_augmentations) | set(args.random_prune_ops)
        )

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

    # And again for the null arm, carrying BOTH the count and the seed. Two
    # draws of the same size at different seeds are different experiments;
    # without the seed they would share a work_dir, interleave their logs, and
    # -- because the Nexus sbatch's resume guard keys off
    # {work_dir}/{exp_name}/last_checkpoint -- silently resume each other.
    if args.random_prune_method == "null":
        suffix = f"nullprune{args.random_prune_count}_s{args.random_prune_seed_used}"
        if suffix not in args.exp_name:
            args.exp_name = args.exp_name + "_" + suffix

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    torch.cuda.device(args.local_rank)

    train(args)

    dist_print("Done.")
