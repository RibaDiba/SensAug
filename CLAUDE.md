# sensaug — Sensitivity-Informed Augmentation for Robust Segmentation

Codebase contact: Laura Zheng <lyzheng@umd.edu>

## What this repo does

Trains segmentation models with adaptive, sensitivity-informed augmentation. The core idea: periodically run a perturbation sensitivity analysis (SA) on the model, then weight training augmentations toward the perturbation types the model is currently worst at.

Built on top of MMSegmentation + MMEngine. Custom components (hooks, loops, dataset classes, transforms) are registered via MMSeg/MMEngine registries.

## Cluster & environment

- Cluster: **Della** (Princeton HPC), SLURM scheduler
- Conda env: `sensaug` (Python 3.10, PyTorch 2.0, MMSeg latest)
- Cluster config: `configs/della.yaml` — set all paths here, not in code
- CUDA module: `cuda/11.7.0`

## Active dataset: Cityscapes

Currently configured in `configs/della.yaml`. Data lives at:
```
data/cityscapes/
  gtFine/          ← extracted (annotations)
  leftImg8bit/     ← NOT YET EXTRACTED (see setup steps below)
  leftImg8bit_trainvaltest.zip
  gtFine_trainvaltest.zip
```

### Cityscapes setup — pending steps

MMSeg expects:
- `leftImg8bit/{train,val,test}/city/*.png` — RGB images
- `gtFine/{train,val,test}/city/*_gtFine_labelTrainIds.png` — train-id labels (19-class)

Both are missing. To complete setup, run **on a compute node** (large disk I/O):

```bash
# 1. Extract images
cd data/cityscapes
unzip leftImg8bit_trainvaltest.zip

# 2. Generate labelTrainIds from polygon JSONs
# From the repo root, using mmseg's conversion script:
python -m mmseg.tools.convert_datasets.cityscapes data/cityscapes --nproc 8

# OR using the cityscapes scripts package:
python -c "
import os, glob
from cityscapesscripts.preparation.createTrainIdLabelImgs import main
os.chdir('data/cityscapes')
main()
"
```

After setup, the directory should look like:
```
data/cityscapes/
  leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png  ...
  gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png  ...
```

## How to train

### Submit to Della (recommended)
```bash
# defaults: aug=none, backbone=pspnet, dataset=cityscapes
sbatch job_scripts/train_della.sbatch

# override positional args
sbatch job_scripts/train_della.sbatch ours segformer cityscapes
```

### Run locally
```bash
python train.py \
  --cluster-config=configs/della.yaml \
  --backbone=pspnet \
  --dataset=cityscapes \
  --aug-type=none \
  --work_dir=./experiments \
  --exp_name=none_pspnet_cityscapes \
  --no-inv-aug
```

### Key train.py flags
| Flag | Values | Notes |
|---|---|---|
| `--aug-type` | `none`, `ours`, `default`, `autoaugment`, `augmix`, `randaugment`, `trivialaugment` | `ours` = sensitivity-informed |
| `--backbone` | `pspnet`, `segformer`, `convnext`, `deeplabv3plus`, `swin`, `mae`, `vit` | must have a config under `sensaug/custom_configs/mmseg/<backbone>/` |
| `--dataset` | keys from `configs/della.yaml` | currently only `cityscapes` on Della |
| `--no-inv-aug` | flag | exclude color/photometric augmentations |
| `--no-warmup` | flag | skip clean-training warmup rounds |
| `--resume` | flag | auto-resume from last checkpoint in work_dir |
| `--round_interval` | int (iters) | the **SA pipeline's clock**. Default `max_iters // 20`. Overrides `schedule.round_interval` |
| `--grad-corr` | flag | enable the **gradient cross-correlation pipeline** (logs the matrix R). Works with ANY `--aug-type`, including `none` |
| `--corr-interval` | int (iters) | the **correlation pipeline's clock**. Default `max_iters // 4`. Overrides `schedule.corr_interval` |
| `--sa_interval` | int | ⚠️ dead flag — parsed but never read. SA-curve recompute is hardcoded to every 6th round in `sensaug/loops.py` |

## The two pipelines (and their two clocks)

Training runs two independent measurements. They share no clock and no hook point — do not couple them.

| Pipeline | Question it answers | Clock | Code |
|---|---|---|---|
| Sensitivity analysis (SA) | which perturbations is the model *worst at*? (weights the training aug PDF) | `round_interval` | `sensaug/loops.py` → `RobustValLoop` |
| Gradient cross-correlation | which perturbations are *redundant with each other*? (the matrix R) | `corr_interval` | `sensaug/hooks/grad_hook.py` → `sensaug/hooks/grad_sens_analysis.py` |

- **SA** runs only under `--aug-type=ours`, from the val loop. The SA *curve* is recomputed every 6th round (hardcoded), so its effective cadence is `6 × round_interval`.
- **Correlation** is opt-in via `--grad-corr` and runs for **any** `--aug-type` — measuring R against an `--aug-type=none` baseline is the control the `ours` number needs. It fires from `after_train_iter`, freezing the model and sweeping the whole clean val set (500 images on Cityscapes) for `d loss / d magnitude` per aug per image. `CollectGradientHook` (priority NORMAL) sweeps; `PerturbationSensitivityAnalysisHookWithGradients` (priority LOW) correlates the sweep it just wrote — the priority ordering is load-bearing. Both are given the same `interval`; the shared gate is `fires_at()` in `grad_hook.py`.

## Known issue: `--aug-type=grad_corr` jobs burn their SLURM walltime on SA round-evals, not training

Diagnosed from `experiments/grad_corr_grad_corr_2_pspnet_cityscapes_4gpu_gradcorr` (job `11809659`, 6h walltime): the job was killed by SLURM at the time limit ([logs/grad_corr_2.11809659.err](logs/grad_corr_2.11809659.err) — `CANCELLED ... DUE TO TIME LIMIT`), having reached only iter 28000/80000 (35%). Not a crash. Breakdown of the 6h: **actual training ≈ 38 min (~10%)**, one correlation sweep ≈ 4 min (~1%), **SA round-evals ≈ 5h20m (~88%)**.

**Root cause:** for `--aug-type=grad_corr`, `train.py` sets `cfg.val_cfg.perturbation_set = "diff"`. Every SA round (`round_interval`, default `max_iters // 20`) once `self.sa_curve` is non-`None`, `RobustValLoop.run()` (`sensaug/loops.py`) calls `generate_pdf_new()` → `test_perturbed_new()`, which rebuilds the val/test dataloader once per `(perturbation, magnitude)` pair and runs a full mIoU eval over it. For the `"diff"` vocabulary this inserts the CPU per-image `Diff*` transforms (`sensaug/dataset/augmentations.py`, `_make_diff_transform`/`DIFF_PERTURBATIONS`) — but `DIFFERENTIABLE_PERTURBATIONS` ops (`sensaug/dataset/differentiable_augmentations.py`) are **GPU-batched-only by design** and 40-150x slower per-image on CPU. `update_sa_curve()` (every 6th round) sweeps this same way. In the diagnosed run, two of these round-evals (iters 20000 and 24000) each took 1h49m–3h11m; ordinary training iterations run at ~0.08s/iter and are not the bottleneck.

**Partial fix already applied (uncommitted, this branch):** `sensaug/loops.py` (`RobustValLoop.run()`, guards around the `apply_random_alpha_training_augmentations`/`apply_random_perturbations_train_dataloader_new` calls) and `train.py` (`build_config`, the `RandomAlphaTrainTransform` `perturbation_set` arg) now force the **train** dataloader to always use `perturbation_set="new"`, never rebuilding it onto `"diff"`. This fixed the training-loop side of the 40-150x slowdown. It does **not** touch the **SA round-eval** side (`test_perturbed_new`/`update_sa_curve`/`generate_pdf_new`, still on `"diff"` when `--aug-type=grad_corr`) — that remains the open bug above.

**Why "just switch the SA loop to `\"new\"` too" doesn't work:** `NEW_PERTURBATIONS` (PascalCase MMSeg transform classes, native per-transform units — used by `--aug-type=ours`) and `DIFFERENTIABLE_PERTURBATIONS` (snake_case keys, magnitudes in `[0,1]` — used by the correlation hook) are **disjoint vocabularies**, different op names and different magnitude scales. `RobustValLoop.publish_corr_magnitudes()` (`sensaug/loops.py:570-571`) already guards against publishing a `"new"`-keyed snapshot for a `"diff"`-vocabulary probe (`if self.perturbation_set != "diff" or not self.pdf_dict: return`), specifically because a mismatched snapshot would make every op in `CollectGradientHook` silently fall back to the fixed reference magnitude (`0.5`, `grad_hook.py`) instead of the SA-informed one. So switching the SA loop's `perturbation_set` to `"new"` would fix the walltime problem but **silently disable `--corr-magnitude-mode mode`'s adaptive probing for the whole run** — functionally identical to always passing `--no-corr-sa`, just via a different code path, and without `--no-corr-sa`'s other simplifications (it skips `RobustValLoop` entirely).

**Not yet decided/implemented — two options on the table:**
1. Skip the diff-vocabulary SA computation outright (mirror the train-loop guard in `RobustValLoop.run()`) and accept fixed-magnitude correlation probing. Simple, but gives up adaptive probe magnitudes — equivalent to running with `--no-corr-sa`, so in practice this may mean just recommending `--no-corr-sa` over a code change.
2. Keep adaptive probing, but rewrite `test_perturbed_new`/`adaptive_sensitivity_analysis_new`'s diff-vocabulary sweep to play `DIFFERENTIABLE_PERTURBATIONS` ops through the GPU-batched path (same mechanism `CollectGradientHook` already uses) instead of the CPU per-image dataloader-rebuild path. Preserves the feature, bigger lift.

**Until this is resolved:** `--aug-type=grad_corr` jobs without `--no-corr-sa` should be launched with a much longer `--time`, or with `--round_interval` raised well above the `max_iters // 20` default to reduce how often the expensive sweep fires.

## Cluster config: `configs/della.yaml`

Parsed by `sensaug/cluster_config.py`. Controls all paths **and both pipeline schedules**.

```yaml
data_root: /projects/PUCHALLA/LLP2024/tumor/data
mmconfig_path: /projects/PUCHALLA/LLP2024/tumor/sensaug/custom_configs/mmseg
primary_metric: mIoU

schedule:                       # both in ITERATIONS; null → default
  round_interval: null          # SA pipeline's clock.          null → max_iters // 20
  corr_interval: null           # correlation pipeline's clock. null → max_iters // 4

datasets:
  cityscapes: cityscapes        # key → subfolder under data_root
supported_backbones:
  - pspnet
  - segformer
  - convnext
  - deeplabv3plus
  - swin
  - mae
  - vit
```

Schedule precedence is **CLI flag > `schedule:` block > default**, resolved by `resolve_interval()` in `train.py`. The `schedule:` block is optional — configs without it still load (`SCHEDULE` becomes `{}`).

To add a new dataset: add an entry under `datasets:` and follow the 3-step process below.

## Adding a new dataset (3 steps from README)

**Step 1** — Implement dataset class in `sensaug/dataset/datasets.py`:
```python
@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(classes=(...), palette=[...])
    def __init__(self, img_suffix=".png", seg_map_suffix="_label.png", **kwargs):
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```

**Step 2** — Create a PSPNet training config (lightest backbone, used as base):
```
sensaug/custom_configs/mmseg/pspnet/pspnet_r18-d8_4xb2-80k_DATASETNAME.py
```
Copy from `pspnet_r18-d8_4xb2-80k_a2i2haze.py` and swap dataset refs. Also create:
```
sensaug/custom_configs/mmseg/_base_/datasets/DATASETNAME.py
```

**Step 3** — Add to `configs/della.yaml`:
```yaml
datasets:
  cityscapes: cityscapes
  DATASETNAME: path/relative/to/data_root
```

## Key files

| File | Purpose |
|---|---|
| `train.py` | Main training entrypoint |
| `test.py` | Robustness testing on OOD datasets |
| `configs/della.yaml` | Della cluster paths/datasets (edit this) |
| `sensaug/cluster_config.py` | Parses the YAML config |
| `sensaug/dataset/datasets.py` | Custom dataset class registrations |
| `sensaug/dataset/augmentations.py` | Custom MMSeg transform registrations |
| `sensaug/dataset/differentiable_augmentations.py` | Autograd-compatible augmentation ops (`DIFFERENTIABLE_PERTURBATIONS`) — what R is computed over |
| `sensaug/hooks/sensitivity_hooks.py` | Legacy SA hooks (**not registered by train.py** — the SA loop does this now) |
| `sensaug/hooks/grad_hook.py` | `CollectGradientHook` — the frozen-frame per-image gradient sweep, and the shared `fires_at()` clock |
| `sensaug/hooks/grad_sens_analysis.py` | `PerturbationSensitivityAnalysisHookWithGradients` — builds the cross-correlation matrix R from the sweep |
| `sensaug/loops.py` | Custom train/val loops (RobustValLoop, etc.) — **the SA pipeline** |
| `sensaug/custom_configs/mmseg/` | Per-backbone MMSeg config files |
| `job_scripts/train_della.sbatch` | Della SLURM job script |

## Experiments output

Saved to `./experiments/<exp_name>/`. Contains checkpoints, tensorboard logs, and a copy of the cluster config used (`seg_config.yaml`).

Per-pipeline logs:

| File | Written by | Contents |
|---|---|---|
| `sa_curve_log.txt` | SA pipeline | JSONL, one SA curve per recompute |
| `perturb_eval.txt` | SA pipeline | JSONL, per-round perturbed eval metrics |
| `aug_gradient_log.txt` | correlation pipeline | JSONL, one record per sweep batch — every per-image gradient, so R is recomputable offline without retraining |
| `corr_matrix_log.json` | correlation pipeline | one JSON array, one record per emission: raw + scale-normalized R, dropped ops, shared-image-factor loadings |
| `corr_bootstrap_log.txt` | correlation pipeline | JSONL, per-cell bootstrap CIs and BH-FDR q-values |

Launch tensorboard:
```bash
tensorboard --logdir experiments/ --host 0.0.0.0
```

## Nexus cluster

Reference config at `configs/nexus.yaml` — shows the full set of supported datasets (cityscapes, ade20k, pascal_voc12, loveda, potsdam, synapse, a2i2haze, acdc, idd). Della currently only has Cityscapes.
