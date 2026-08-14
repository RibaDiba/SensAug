#!/usr/bin/env python
"""Compute the gradient cross-correlation matrix R for an already-trained
checkpoint, without retraining.

CollectGradientHook / PerturbationSensitivityAnalysisHookWithGradients
(sensaug/hooks/grad_hook.py, sensaug/hooks/grad_sens_analysis.py) normally
fire from `after_train_iter` during an `--aug-type=grad_corr` training run, but their
actual work lives in plain methods -- `before_run`/`_sweep`/`_emit` -- that
only need a loaded model and the val dataloader, never the optimizer or a
live training loop. This script builds a Runner from a finished experiment's
dumped config, loads the checkpoint's weights directly (skipping
`.train()`/`.test()`/`.resume()`, which would require building an unused
train loop), and calls those hook methods once for that checkpoint.

Output lands in the same four files the live pipeline writes --
aug_gradient_log.txt, corr_matrix_log.json, corr_bootstrap_log.txt and
corr_redundancy_log.txt -- in the experiment's own work_dir by default, so it
is recomputable/comparable alongside anything a live `--aug-type=grad_corr` run already
logged there. The redundancy score is written but goes nowhere else: nothing
here reweights a pdf, since there is no training loop to reweight one for.

Usage:
    python scripts/compute_grad_corr.py --work-dir experiments/none_pspnet_cityscapes
"""

import argparse
import glob
import os
import sys

# Repo root, so `sensaug` resolves whether this is run as
# `python scripts/compute_grad_corr.py` from the repo root or from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mmengine.config import Config
from mmengine.runner import Runner

# torch >= 2.6 defaults torch.load to weights_only=True, which cannot
# unpickle this repo's checkpoints (message_hub carries mmengine
# HistoryBuffer objects, themselves holding numpy arrays -- both rejected by
# the restricted unpickler one class at a time). mmengine's own checkpoint
# loader (Runner.load_checkpoint -> torch.load) does not expose a
# weights_only override, so there is no way to ask it for the old behavior
# through its public API. These are first-party checkpoints this repo's own
# training loop just wrote, not untrusted input, so restore the pre-2.6
# default for this process only.
_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

import sensaug.dataset.datasets  # noqa: F401  registers custom DATASETS
from sensaug.hooks import *  # noqa: F401,F403  registers custom HOOKS
from sensaug.visualizer import BPSegLocalVisualizer  # noqa: F401
from sensaug.dataset.idbh import IDBHTransform  # noqa: F401

from sensaug.hooks.grad_hook import CollectGradientHook
from sensaug.hooks.grad_sens_analysis import (
    PerturbationSensitivityAnalysisHookWithGradients,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the gradient cross-correlation matrix R for an "
            "already-trained checkpoint, without retraining."
        )
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="finished experiment directory (holds the dumped config + checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="explicit checkpoint path; default is the best*.pth in --work-dir",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        default=False,
        help="use the checkpoint pointed to by last_checkpoint instead of best*.pth",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "where to write aug_gradient_log.txt / corr_matrix_log.json / "
            "corr_bootstrap_log.txt; default is --work-dir itself"
        ),
    )
    parser.add_argument("--sweep-batch-size", type=int, default=1)
    parser.add_argument(
        "--magnitude-mode",
        default="mode",
        choices=["mode", "sampled_shared", "sampled_independent", "fixed"],
    )
    parser.add_argument(
        "--magnitudes-path",
        default=None,
        help=(
            "corr_magnitudes.json-shaped seed file (a live run's own log, or "
            "the output of scripts/calibrate_kid_magnitudes.py); default is "
            "<work-dir>/corr_magnitudes.json if it exists, else the fixed "
            "ref-magnitude fallback"
        ),
    )
    parser.add_argument("--ref-magnitude", type=float, default=0.5)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--n-min", type=int, default=100)
    parser.add_argument("--no-bootstrap", action="store_true", default=False)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    return parser.parse_args()


def resolve_checkpoint(work_dir, checkpoint, use_latest):
    if checkpoint is not None:
        return checkpoint
    if use_latest:
        last_checkpoint_f = os.path.join(work_dir, "last_checkpoint")
        with open(last_checkpoint_f) as f:
            name = os.path.basename(f.read().strip())
        return os.path.join(work_dir, name)
    best = glob.glob(os.path.join(work_dir, "best*.pth"))
    if not best:
        raise FileNotFoundError(
            f"no best*.pth checkpoint found in {work_dir}; pass --checkpoint "
            f"or --use-latest"
        )
    return best[0]


class _PosthocRunner:
    """Forwards everything to a real mmengine Runner except `.iter` /
    `.max_iters`, which the real Runner derives from a train_loop we
    deliberately never build (building one would need an optimizer and the
    train dataloader just to log a checkpoint fraction). Free attribute
    assignment lets the grad-corr hooks stash `aug_grad_buffer` /
    `corr_magnitudes` / `aug_grad_magnitude_info` directly, the same way they
    do on a real Runner mid-training.
    """

    def __init__(self, runner, iter_, max_iters):
        self._runner = runner
        self.iter = iter_
        self.max_iters = max_iters

    def __getattr__(self, name):
        return getattr(self._runner, name)


def main():
    args = parse_args()
    work_dir = os.path.abspath(args.work_dir)

    config_paths = glob.glob(os.path.join(work_dir, "*.py"))
    if not config_paths:
        raise FileNotFoundError(f"no dumped config (*.py) found in {work_dir}")
    cfg = Config.fromfile(config_paths[0])

    checkpoint_path = resolve_checkpoint(work_dir, args.checkpoint, args.use_latest)
    # cfg.train_cfg can be None here (e.g. a work_dir whose dumped *.py is
    # itself a previous test-only/post-hoc dump rather than the original
    # training config) -- max_iters is recovered below from the checkpoint's
    # own embedded config in that case.
    train_cfg = cfg.get("train_cfg")
    max_iters = train_cfg.max_iters if train_cfg else None

    # Training-only state: never build it, so we never need an optimizer or
    # the train dataloader just to sweep the (already-clean) val set.
    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    # Avoid double-registering this experiment's own dumped grad-corr hooks
    # (present verbatim if it already trained with --aug-type=grad_corr) alongside the
    # ones this script instantiates itself below.
    cfg.custom_hooks = None
    cfg.resume = False
    # The dumped config's launcher is whatever the original training run
    # used (e.g. 'pytorch' for a multi-GPU job); Runner.from_cfg would try to
    # join a distributed process group using torchrun env vars that are not
    # set here.
    cfg.launcher = "none"
    cfg.load_from = checkpoint_path
    cfg.work_dir = os.path.abspath(args.output_dir) if args.output_dir else work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    # Not .train()/.test()/.resume(): those need a fully built train loop.
    # load_checkpoint() puts the weights on the model and hands back the
    # checkpoint dict, which is all this script needs.
    ckpt = runner.load_checkpoint(cfg.load_from)
    loaded_iter = int(ckpt["meta"]["iter"])

    if max_iters is None and ckpt["meta"].get("cfg"):
        ckpt_cfg = Config.fromstring(ckpt["meta"]["cfg"], file_format=".py")
        ckpt_train_cfg = ckpt_cfg.get("train_cfg")
        if ckpt_train_cfg:
            max_iters = ckpt_train_cfg.max_iters

    proxy = _PosthocRunner(runner, iter_=loaded_iter - 1, max_iters=max_iters)

    magnitudes_path = args.magnitudes_path
    if magnitudes_path is None:
        default_path = os.path.join(work_dir, "corr_magnitudes.json")
        if os.path.exists(default_path):
            magnitudes_path = default_path

    collect_hook = CollectGradientHook(
        interval=1,
        sweep_batch_size=args.sweep_batch_size,
        ref_magnitude=args.ref_magnitude,
        probe_seed=args.probe_seed,
        magnitude_mode=args.magnitude_mode,
        magnitudes_path=magnitudes_path,
    )
    analysis_hook = PerturbationSensitivityAnalysisHookWithGradients(
        interval=1,
        n_min=args.n_min,
        bootstrap=not args.no_bootstrap,
        bootstrap_reps=args.bootstrap_reps,
    )

    checkpoint_frac = loaded_iter / max_iters if max_iters else 0.0

    # One sweep, one emission -- this deliberately bypasses
    # after_train_iter/fires_at, which exist to gate a *repeating* schedule
    # against runner.iter. A post-hoc run only ever wants exactly one.
    collect_hook.before_run(proxy)
    collect_hook._sweep(proxy, checkpoint_frac)
    analysis_hook._emit(proxy, checkpoint_frac, proxy.aug_grad_buffer)
    collect_hook.after_run(proxy)

    print(f"checkpoint: {checkpoint_path} (iter {loaded_iter}/{max_iters})")
    print(f"wrote: {os.path.join(cfg.work_dir, 'aug_gradient_log.txt')}")
    print(f"wrote: {os.path.join(cfg.work_dir, 'corr_matrix_log.json')}")
    if not args.no_bootstrap:
        print(f"wrote: {os.path.join(cfg.work_dir, 'corr_bootstrap_log.txt')}")


if __name__ == "__main__":
    main()
