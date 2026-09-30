#!/usr/bin/env python
"""Post-training sensitivity/robustness ranking for one checkpoint, over EVERY op.

Reproduces sensaug's own SA methodology -- NOT a fixed magnitude grid:

  1. RobustValLoop.update_sa_curve()  -> adaptive_sensitivity_analysis_new():
     per op, adaptively fit a sensitivity curve (mIoU-drop + level + KID via
     objective_function) with PCHIP interpolation and pull out `num_levels`
     representative magnitudes. ~3-6 real evals/op at the SubsetTestLoop ratio,
     not a linspace.
  2. RobustValLoop.test_perturbed_new(): eval each op at those adaptive levels.
  3. RobustValLoop.generate_pdf_new(): RANK each op's levels by mIoU (worst
     first) and weight them with betabinom.pmf(rank, n, 0.75, 1.0); the per-op
     probability mass is that op's "relevance" -- exactly the signal mRMR reads.

The only deviations from the trained config: `pruned_augmentations` is cleared
and `remove_H` is forced off, so a statically-pruned run still gets a full 32-op
ranking (the whole reason this script exists -- `perturb_eval.txt` has nothing
for the pruned ops).

Config is recovered from the checkpoint's own `meta["cfg"]`; the work_dir's
dumped *.py is NOT trusted (an earlier run of this script, or test.py, may have
overwritten it -- mmengine's Runner.__init__ calls dump_config()). Results are
written to <work-dir>/augmentation_sensitivity.json; the Runner's own work_dir is
pointed at a scratch dir so nothing in the experiment dir is touched.

Usage:
    python scripts/eval_all_augmentations.py --work-dir experiments/<exp>
"""

import argparse
import csv
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# torch >= 2.6 defaults torch.load to weights_only=True, which cannot unpickle
# this repo's checkpoints. Same shim as scripts/compute_grad_corr.py.
_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from mmengine.runner import Runner

import sensaug.dataset.datasets  # noqa: F401  registers DATASETS
from sensaug.hooks import *  # noqa: F401,F403  registers HOOKS
from sensaug.visualizer import BPSegLocalVisualizer  # noqa: F401
from sensaug.dataset.idbh import IDBHTransform  # noqa: F401
from sensaug.loops import *  # noqa: F401,F403  registers LOOPS (RobustValLoop, SubsetTestLoop)
from sensaug.dataset.gpu_augment import GpuAugSegDataPreProcessor  # noqa: F401
from sensaug.dataset.augmentations import resolve_perturbation_set

_METRICS = ("mIoU", "aAcc", "mAcc")

_DATASET_STEM = {
    "PascalVOCDataset": "voc",
    "CityscapesDataset": "cityscapes",
    "ADE20KDataset": "ade20k",
    "LoveDADataset": "loveda",
    "PotsdamDataset": "potsdam",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", required=True, help="finished experiment dir")
    p.add_argument("--checkpoint", default=None, help="explicit ckpt path")
    p.add_argument(
        "--use-latest",
        action="store_true",
        help="use last_checkpoint instead of best*.pth",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="SubsetTestLoop / round-eval fraction; default = trained val_cfg.ratio",
    )
    p.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="representative magnitudes per op for the adaptive SA curve",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="adaptive-search stopping tolerance (adaptive_sensitivity_analysis_new)",
    )
    p.add_argument(
        "--sa-curve",
        default=None,
        help="skip the adaptive search; load this {op:[levels]} JSON instead",
    )
    p.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="restrict to these op names (smoke runs); default = all in the set",
    )
    p.add_argument("--out", default=None, help="output JSON (default <work-dir>/augmentation_sensitivity.json)")
    p.add_argument("--csv", dest="csv", action="store_true", default=True)
    p.add_argument("--no-csv", dest="csv", action="store_false")
    p.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    return args


def resolve_checkpoint(work_dir, checkpoint, use_latest):
    if checkpoint is not None:
        return checkpoint
    if use_latest:
        with open(os.path.join(work_dir, "last_checkpoint")) as f:
            return os.path.join(work_dir, os.path.basename(f.read().strip()))
    best = glob.glob(os.path.join(work_dir, "best*.pth"))
    if not best:
        raise FileNotFoundError(
            f"no best*.pth in {work_dir}; pass --checkpoint or --use-latest"
        )
    return best[0]


def recover_config(work_dir, checkpoint_path):
    """Prefer the checkpoint's embedded config; the work_dir *.py may have been
    overwritten by an earlier eval/test run (mmengine dump_config())."""
    ckpt = _torch_load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    cfg_str = meta.get("cfg")
    if cfg_str and "RobustValLoop" in cfg_str:
        return Config.fromstring(cfg_str, file_format=".py"), int(meta.get("iter", 0))
    py = glob.glob(os.path.join(work_dir, "*.py"))
    if not py:
        raise FileNotFoundError(
            f"no usable config: checkpoint meta lacks one and no *.py in {work_dir}"
        )
    print_log(
        f"WARNING: checkpoint meta has no RobustValLoop config; falling back to "
        f"{py[0]} (may be a stale eval dump)",
        logger="current",
        level=30,
    )
    return Config.fromfile(py[0]), int(meta.get("iter", 0))


def per_op_key(op, metric):
    return f"{op.replace('_', '')}_{metric}"


def build_table(sa_curve, miou_record, pdf_dict, final_metrics, clean, trained_pruned):
    """One row per op: SA relevance (summed pdf mass), mean metrics over its
    adaptive levels, worst level, and whether the trained run had pruned it."""
    # relevance = total probability the ranked pdf puts on this op
    relevance = {}
    for (op, _lvl), prob in pdf_dict.items():
        if op == "none":
            continue
        relevance[op] = relevance.get(op, 0.0) + float(prob)

    rows = {}
    for op, levels in sa_curve.items():
        lvl_rows = []
        for lvl in levels:
            m = miou_record.get(op, {}).get(lvl)
            if m is not None:
                lvl_rows.append((float(lvl), float(m), float(pdf_dict.get((op, lvl), 0.0))))
        lvl_rows.sort(key=lambda t: t[1])  # worst mIoU first
        rows[op] = {
            "relevance": relevance.get(op, 0.0),
            "mean_mIoU": final_metrics.get(per_op_key(op, "mIoU")),
            "mean_aAcc": final_metrics.get(per_op_key(op, "aAcc")),
            "mean_mAcc": final_metrics.get(per_op_key(op, "mAcc")),
            "worst_level": lvl_rows[0][0] if lvl_rows else None,
            "worst_mIoU": lvl_rows[0][1] if lvl_rows else None,
            "levels": [
                {"level": lv, "mIoU": mi, "prob": pr} for lv, mi, pr in lvl_rows
            ],
            "trained_pruned": op in trained_pruned,
        }
    return rows


def print_summary(clean, rows, trained_pruned):
    base = clean.get("mIoU", float("nan"))
    order = sorted(
        rows.items(),
        key=lambda kv: (kv[1]["relevance"] if kv[1]["relevance"] is not None else 0.0),
        reverse=True,
    )
    print(f"\nclean mIoU = {base:.2f}   (ratio-subset)\n")
    print(f"{'op':<18}{'relevance':>10}{'mean mIoU':>11}{'Δ clean':>9}{'worst@':>8}{'':>4}")
    print("-" * 62)
    for op, r in order:
        mm = r["mean_mIoU"]
        mm = float("nan") if mm is None else mm
        wl = r["worst_level"]
        wl = float("nan") if wl is None else wl
        mark = "  <-- trained-pruned" if r["trained_pruned"] else ""
        print(
            f"{op:<18}{r['relevance']:>10.4f}{mm:>11.2f}{mm - base:>9.2f}{wl:>8.2f}{mark}"
        )


def write_csv(path, clean, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "level", "mIoU", "prob", "relevance", "mean_mIoU", "trained_pruned"])
        w.writerow(["__clean__", "", clean.get("mIoU", ""), "", "", "", ""])
        for op, r in rows.items():
            for lv in r["levels"]:
                w.writerow(
                    [op, lv["level"], lv["mIoU"], lv["prob"], r["relevance"],
                     r["mean_mIoU"], r["trained_pruned"]]
                )


def main():
    args = parse_args()
    work_dir = os.path.abspath(args.work_dir)

    checkpoint_path = resolve_checkpoint(work_dir, args.checkpoint, args.use_latest)
    cfg, loaded_iter = recover_config(work_dir, checkpoint_path)

    trained_val = dict(cfg.get("val_cfg") or {})
    trained_pruned = set(trained_val.get("pruned_augmentations") or [])
    ratio = args.ratio if args.ratio is not None else float(trained_val.get("ratio", 0.25))
    pert_set = trained_val.get("perturbation_set", "diff32")

    # scratch work_dir: Runner.__init__ -> dump_config() writes there; must NOT be
    # the experiment dir (that is what clobbered the training config dumps before).
    scratch = tempfile.mkdtemp(prefix="augeval_")

    # kill training-only state
    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    cfg.custom_hooks = None
    cfg.resume = False
    cfg.launcher = args.launcher
    cfg.load_from = checkpoint_path
    cfg.work_dir = scratch

    # SubsetTestLoop is required by adaptive_sensitivity_analysis_new
    # (calculate_miou_kid_new asserts cfg.test_cfg.type == "SubsetTestLoop").
    cfg.test_cfg = dict(type="SubsetTestLoop", ratio=ratio)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    # RobustValLoop, but UNFILTERED: clear the static prune and remove_H so the
    # ranking covers every op the run could have sampled.
    vc = dict(trained_val)
    if trained_val.get("type") not in ("RobustValLoop", "GradCorrValLoop"):
        # A run with no SA loop of its own (--aug-type=none, grad_corr
        # --no-corr-sa, ...) saved mmengine's stock ValLoop config, which carries
        # none of RobustValLoop's kwargs. Fill in the two it cannot run without,
        # and put the model on the GPU preprocessor the diff32 eval path drives.
        # The preprocessor holds no weights, so the checkpoint loads unchanged.
        print_log(
            f"trained val loop was {trained_val.get('type')!r} (no SA loop); "
            f"evaluating it on the {pert_set} set via GpuAugSegDataPreProcessor",
            logger="current",
        )
        vc = {"sa_curve_path": "sensaug/testing/shared_levels_diff32.json"}
        cfg.model.data_preprocessor.type = "GpuAugSegDataPreProcessor"
        cfg.model.data_preprocessor.pruned_ops = []
    vc["perturbation_set"] = pert_set
    vc["type"] = "RobustValLoop"
    vc["pruned_augmentations"] = []
    vc["remove_H"] = False
    vc["ratio"] = ratio
    vc["random_aug"] = False
    vc["uniform"] = False
    vc["weighted_augs"] = False
    if args.ops:
        keep = set(args.ops)
        allops = list(resolve_perturbation_set(pert_set).keys())
        vc["pruned_augmentations"] = [o for o in allops if o not in keep]
    cfg.val_cfg = vc

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(cfg.load_from)
    # SubsetTestLoop.run() (used by adaptive_sensitivity_analysis_new) fires
    # after_test_epoch -> LoggerHook, which needs json_log_path from
    # LoggerHook.before_run(). Calling test_loop.run() directly never fires
    # before_run (only Runner.train/val/test do), so fire it once here.
    runner.call_hook("before_run")

    # RobustValLoop.__init__ reads runner.cfg.train_cfg.val_interval (and
    # runner.iter, which is 0 without a train loop -> n_rounds 0, harmless
    # because we call the loop's methods directly, not run()).
    runner.cfg.train_cfg = Config(dict(val_interval=1000, max_iters=max(loaded_iter, 1)))

    loop = runner.val_loop  # builds the unfiltered RobustValLoop

    # capture test_perturbed_new's miou_record when generate_pdf_new calls it
    _orig_tpn = loop.test_perturbed_new
    _stash = {}

    def _tpn():
        r = _orig_tpn()
        _stash["miou_record"], _stash["final_metrics"] = r
        return r

    loop.test_perturbed_new = _tpn

    # ---- clean baseline (SubsetTestLoop, same ratio + KID as the SA search) ----
    from sensaug.runner_utils import apply_perturbations_dataloader

    apply_perturbations_dataloader(runner, train=False, perturb_levels={}, perturbation_set=pert_set)
    clean_metrics = runner.test_loop.run()
    clean = {k: float(clean_metrics[k]) for k in _METRICS if k in clean_metrics}

    # ---- SA ranking ----------------------------------------------------------
    if args.sa_curve:
        loop.static_sa_curve_path = args.sa_curve
        loop.load_sa_curve()
        print_log(f"loaded SA curve from {args.sa_curve}", logger="current")
    else:
        print_log(
            f"running adaptive SA (num_levels={args.num_levels}, tol={args.tolerance}, "
            f"ratio={ratio}) over the full {pert_set} vocabulary...",
            logger="current",
        )
        # update_sa_curve() hardcodes num_levels=5 / tolerance=0.05; call the
        # underlying routine directly so --num-levels / --tolerance take effect.
        from copy import deepcopy
        from sensaug.sensitivity_analysis import adaptive_sensitivity_analysis_new

        loop.sa_curve = adaptive_sensitivity_analysis_new(
            deepcopy(runner.cfg.val_dataloader),
            runner,
            num_levels=args.num_levels,
            tolerance=args.tolerance,
            perturbation_set=loop.perturbation_set,
            exclude=loop.pruned_augmentations,  # [] unless --ops
        )

    # generate_pdf_new(): evals the adaptive levels (via the wrapped
    # test_perturbed_new) and ranks them per op with betabinom weights.
    pdf_dict, final_metrics = loop.generate_pdf_new()
    miou_record = _stash.get("miou_record", {})

    runner.call_hook("after_run")

    if not is_main_process():
        return

    rows = build_table(loop.sa_curve, miou_record, pdf_dict, final_metrics, clean, trained_pruned)

    result = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "iter": loaded_iter,
        "dataset": _DATASET_STEM.get(cfg.get("dataset_type"), cfg.get("dataset_type", "unknown")),
        "perturbation_set": pert_set,
        "ratio": ratio,
        "num_levels": args.num_levels,
        "tolerance": args.tolerance,
        "method": "sensaug adaptive SA + betabinom-ranked pdf (pruned_augmentations & remove_H cleared)",
        "trained_pruned_augmentations": sorted(trained_pruned),
        "clean": clean,
        "sa_curve": {op: [float(x) for x in lv] for op, lv in loop.sa_curve.items()},
        "miou_record": {
            op: {str(k): float(v) for k, v in lv.items()} for op, lv in miou_record.items()
        },
        "pdf": {f"{op}|{lvl}": float(pr) for (op, lvl), pr in pdf_dict.items()},
        "per_op": rows,
    }

    out_path = args.out or os.path.join(work_dir, "augmentation_sensitivity.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print_log(f"wrote {out_path}", logger="current")
    if args.csv:
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        write_csv(csv_path, clean, rows)
        print_log(f"wrote {csv_path}", logger="current")

    print_summary(clean, rows, trained_pruned)


if __name__ == "__main__":
    main()
