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
- **Compute nodes have no internet access.** Backbones whose config downloads a
  pretrained checkpoint at `model.init_weights()` time (segformer, pspnet-rsb,
  convnext, swin) need that checkpoint pre-staged — see
  [Pretrained backbone checkpoints](#pretrained-backbone-checkpoints) below.

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

### Pretrained backbone checkpoints

`segformer`, `pspnet`'s RSB variants, `convnext`, and `swin` configs point
`model.backbone.init_cfg` at a `download.openmmlab.com` URL, fetched at
`model.init_weights()` time. On a compute node (no internet) this crashes with
`socket.gaierror: Name or service not known` — this is exactly what killed the
`default_segformer_1` / `grad_corr_segformer_1` jobs on 2026-08-10.

`train.py`'s `build_config()` now redirects any such URL to
`<pretrained_cache_dir>/<basename(url)>` (key set in `configs/della.yaml`) and
raises a clear `FileNotFoundError` naming the exact fetch command if it isn't
cached yet, instead of the DNS traceback. Populate the cache once, from a Della
**login** node or your own machine + `rsync`/`scp` (not a compute node / sbatch
job):

```bash
python scripts/download_pretrained_checkpoints.py --backbone segformer
# or: --backbone pspnet convnext swin, or --all
```

Not covered by this mechanism: `deeplabv3plus` (uses mmcv's
`pretrained='open-mmlab://resnet50_v1c'` shorthand, not a literal URL) and
`mae`/`vit` (different init path). No failure has been observed there yet.

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
| `--aug-type` | `none`, `ours`, `default`, `grad_corr`, `random`, `autoaugment`, `augmix`, `randaugment`, `trivialaugment`, `idbh`, `vip` | `ours` = sensitivity-informed. `grad_corr` = **enables the gradient cross-correlation pipeline** (logs the matrix R) — it is its own `--aug-type` value, not a separate flag. (A standalone `--grad-corr` flag existed at one point; it's gone — `--aug-type=grad_corr` is the only way to turn the correlation pipeline on.) |
| `--backbone` | `pspnet`, `segformer`, `convnext`, `deeplabv3plus`, `swin`, `mae`, `vit` | must have a config under `sensaug/custom_configs/mmseg/<backbone>/` |
| `--dataset` | keys from `configs/della.yaml` | currently only `cityscapes` on Della |
| `--no-inv-aug` | flag | exclude color/photometric augmentations |
| `--no-warmup` | flag | skip clean-training warmup rounds |
| `--no-corr-sa` | flag | under `--aug-type=grad_corr`, disable the SA loop — trains exactly like `none` while still running the correlation measurement. This is the control arm |
| `--resume` | flag | auto-resume from last checkpoint in work_dir |
| `--round_interval` | int (iters) | the **SA pipeline's clock**. Default `max_iters // 20`. Overrides `schedule.round_interval` |
| `--corr-interval` | int (iters) | the **correlation pipeline's clock**. Only meaningful under `--aug-type=grad_corr`. Default `max_iters // 4`. Overrides `schedule.corr_interval` |
| `--corr-lambda` | float | redundancy down-weighting strength (**Lever 3**). `0` (default) leaves the pdf bit-identical — the control arm. See below |
| `--corr-red-mode` | `squared`, `abs`, `signed` | how a row of R reduces to one score per op. `squared` default |
| `--corr-lambda-ramp` | `linear`, `constant` | ramp λ from 0 over training (default) vs. full strength from the first R emission |
| `--corr-keep-within-op` | flag | keep the `lighter_X`/`darker_X` and `_pos`/`_neg` cells in `red(a)`; excluded by default |
| `--sa_interval` | int | ⚠️ dead flag — parsed but never read. SA-curve recompute is hardcoded to every 6th round in `sensaug/loops.py` |

## The two pipelines (and their two clocks)

Training runs two independent measurements. They share no clock and no hook point — do not couple them.

| Pipeline | Question it answers | Clock | Code |
|---|---|---|---|
| Sensitivity analysis (SA) | which perturbations is the model *worst at*? (weights the training aug PDF) | `round_interval` | `sensaug/loops.py` → `RobustValLoop` |
| Gradient cross-correlation | which perturbations are *redundant with each other*? (the matrix R) | `corr_interval` | `sensaug/hooks/grad_hook.py` → `sensaug/hooks/grad_sens_analysis.py` |

- **SA** runs only under `--aug-type=ours`, from the val loop. The SA *curve* is recomputed every 6th round (hardcoded), so its effective cadence is `6 × round_interval`.
- **Correlation** is opt-in via `--aug-type=grad_corr` — it is its own `--aug-type` value, not a flag layered on top of another one (`--aug-type=none --grad-corr` is not valid; that flag doesn't exist). It fires from `after_train_iter`, freezing the model and sweeping the whole clean val set (500 images on Cityscapes) for `d loss / d magnitude` per aug per image. `CollectGradientHook` (priority NORMAL) sweeps; `PerturbationSensitivityAnalysisHookWithGradients` (priority LOW) correlates the sweep it just wrote — the priority ordering is load-bearing. Both are given the same `interval`; the shared gate is `fires_at()` in `grad_hook.py`. For the unaugmented control arm — R measured against a baseline with no training augmentation, which is what the SA-on number gets compared to — pass `--aug-type=grad_corr --no-corr-sa`: that disables the SA loop, so the run trains exactly like `none` while still running the correlation measurement.

## The three augmentation vocabularies

Everything above turns on which set of op names is in play. There are three, and
two of them share all 32 keys while meaning different classes under each.

| `perturbation_set` | ops | keys | implementation | used by |
|---|---|---|---|---|
| `new` | 20 | PascalCase (`BrightnessTransform`, `NegativeRotate`) | CPU cv2/numpy transforms | `--aug-type=ours`, `random` |
| `diff` | 32 | snake_case (`lighter_R`, `rotate_neg`) | GPU-batched torch ops, wrapped per-image | **measurement only** — the gradient probe |
| `aligned` | 32 | snake_case, **identical to `diff`** | the CPU transform classes | `--aug-type=grad_corr` |

`resolve_perturbation_set()` (`sensaug/dataset/augmentations.py`) is the single
dispatch point. An import-time assertion pins
`set(ALIGNED_PERTURBATIONS) == set(DIFFERENTIABLE_PERTURBATIONS)`.

**Why `aligned` exists.** R is indexed by the 32 `DIFFERENTIABLE_PERTURBATIONS`
names. The training pdf is indexed by whatever `perturbation_set` resolves to. Under
`new` those two overlap on *nothing*, so no per-op quantity read off R could ever
index into the pdf. `diff` has the right names but is GPU-batched-only by design and
40–150× slower applied per-image on CPU, which is exactly what a training pipeline
and the SA round-eval both do. `aligned` is the same 32 names through the fast CPU
classes.

Two things `aligned` does **not** give you:

- **Magnitude scales are not calibrated against `diff`.** Same op identity, different
  units — `blur` is the starkest (cv2's kernel-size-derived implicit sigma vs. sigma
  directly). A per-op *score* transfers between the two; a magnitude does not. This
  degrades the SA→probe magnitude handoff, not the op identity.
- **The 10 geometric ops' R is still contaminated.** The CPU classes warp the label
  with the image (`Rotate._rotate_seg`), so training on them is fine — but
  `_DiffAugTransform.transform` and `CollectGradientHook._grad_for_op` do not, so
  their measured `dL/dmagnitude` is dominated by image-label misalignment. Measured
  consequence: the geometric ops correlate with almost nothing (mean |r| 0.08–0.11 vs
  0.14–0.20 photometric), so `red(a)` ranks them *least* redundant and would
  **upweight** them. Until `warp_image_and_label` is wired into those two call sites,
  prefer `--photometric-only` for any Lever 3 run you intend to report.

Because `perturbation_set` selects between two registries with identical keys,
`_perturbation_transform_cfg()` (`sensaug/runner_utils.py`) takes it as an argument.
Omitting it falls back to name-based dispatch, which resolves the 32 shared keys to
the **`diff`** classes — the slow path — without failing or logging anything.

## Lever 3: redundancy down-weighting of the training pdf

`q(a) ∝ pdf_old(a) · exp(−λ · red(a))` — the closed-form solution to minimising
`KL(q ‖ pdf_old)` subject to a budget on `Σ q(a)·red(a)`. `red(a)` is the
standardized row sum of R. Lives in `sensaug/redundancy.py` (pure numpy, no mmseg).

- **Handoff.** `PerturbationSensitivityAnalysisHookWithGradients.prune_augmentations`
  (which no longer prunes) publishes `runner.corr_redundancy` and appends to
  `corr_redundancy_log.txt`; `RobustValLoop._apply_redundancy_reweighting` reads
  whatever is current. The hook fires on `corr_interval`, the loop on
  `round_interval` — they never have to agree, and before the first R emission the
  pdf is untouched.
- **λ is portable because `red(a)` is standardized.** Verified against all four
  logged `corr_matrix_log.json` files (10 checkpoints, both A=14 and A=32):
  λ=0.1 → 1.4–1.6×, **λ=0.25 → 2.3–3.1×**, λ=0.5 → 5.2–9.7×, λ=1.0 → 27–94×.
  Portability itself degrades past λ≈0.5. Recompute any time with
  `python scripts/calibrate_lambda.py` (offline, no GPU).
- **λ=0 is bit-identical**, not merely close — it short-circuits. That is what makes
  it usable as the control arm.
- **`("none", 0)` is held fixed.** Only the perturbation mass is redistributed, so
  Lever 3 changes *which* augmentation is sampled, never *how often* augmentation
  happens.
- **Three guards, all of which log rather than degrade silently:** the score is
  withheld when the shared-factor loading exceeds `_LOADING_ALARM` (0.9) — R is then
  ranking which *images* are hard; when no cell survives the FDR gate; and when
  `red(a)` has near-zero variance.
- Down-weighting is soft and structurally cannot reach zero. At the correlation sizes
  actually observed (mean |r| 0.11–0.22) deletion would not be justified.

## Known issue: `--aug-type=grad_corr` jobs burn their SLURM walltime on SA round-evals, not training

Diagnosed from `experiments/grad_corr_grad_corr_2_pspnet_cityscapes_4gpu_gradcorr` (job `11809659`, 6h walltime): the job was killed by SLURM at the time limit ([logs/grad_corr_2.11809659.err](logs/grad_corr_2.11809659.err) — `CANCELLED ... DUE TO TIME LIMIT`), having reached only iter 28000/80000 (35%). Not a crash. Breakdown of the 6h: **actual training ≈ 38 min (~10%)**, one correlation sweep ≈ 4 min (~1%), **SA round-evals ≈ 5h20m (~88%)**.

**Root cause:** for `--aug-type=grad_corr`, `train.py` sets `cfg.val_cfg.perturbation_set = "diff"`. Every SA round (`round_interval`, default `max_iters // 20`) once `self.sa_curve` is non-`None`, `RobustValLoop.run()` (`sensaug/loops.py`) calls `generate_pdf_new()` → `test_perturbed_new()`, which rebuilds the val/test dataloader once per `(perturbation, magnitude)` pair and runs a full mIoU eval over it. For the `"diff"` vocabulary this inserts the CPU per-image `Diff*` transforms (`sensaug/dataset/augmentations.py`, `_make_diff_transform`/`DIFF_PERTURBATIONS`) — but `DIFFERENTIABLE_PERTURBATIONS` ops (`sensaug/dataset/differentiable_augmentations.py`) are **GPU-batched-only by design** and 40-150x slower per-image on CPU. `update_sa_curve()` (every 6th round) sweeps this same way. In the diagnosed run, two of these round-evals (iters 20000 and 24000) each took 1h49m–3h11m; ordinary training iterations run at ~0.08s/iter and are not the bottleneck.

**Partial fix already applied (uncommitted, this branch):** `sensaug/loops.py` (`RobustValLoop.run()`, guards around the `apply_random_alpha_training_augmentations`/`apply_random_perturbations_train_dataloader_new` calls) and `train.py` (`build_config`, the `RandomAlphaTrainTransform` `perturbation_set` arg) now force the **train** dataloader to always use `perturbation_set="new"`, never rebuilding it onto `"diff"`. This fixed the training-loop side of the 40-150x slowdown. It does **not** touch the **SA round-eval** side (`test_perturbed_new`/`update_sa_curve`/`generate_pdf_new`, still on `"diff"` when `--aug-type=grad_corr`) — that remains the open bug above.

**Why "just switch the SA loop to `\"new\"` too" doesn't work:** `NEW_PERTURBATIONS` and `DIFFERENTIABLE_PERTURBATIONS` are **disjoint vocabularies** — different op names and different magnitude scales. `RobustValLoop.publish_corr_magnitudes()` guards against publishing a `"new"`-keyed snapshot for an R-keyed probe, specifically because a mismatched snapshot would make every op in `CollectGradientHook` silently fall back to the fixed reference magnitude (`0.5`, `grad_hook.py`) instead of the SA-informed one. Switching the SA loop to `"new"` would fix the walltime problem but **silently disable `--corr-magnitude-mode mode`'s adaptive probing for the whole run**.

**Fix applied (this branch): `--aug-type=grad_corr` now uses `perturbation_set="aligned"`,** the third vocabulary described above — the same 32 op names, played through the fast CPU transform classes. This keeps the names R needs (so `publish_corr_magnitudes` still fires and adaptive probing survives) while taking the per-image torch ops off both the SA round-eval and the training pipeline. It also un-gates the training-side pdf, which is what makes Lever 3 possible at all.

**Still to be measured, and this is the number that closes this entry out:** the round-eval is now 32 ops × 4 levels instead of 20 × 4 — ~1.6× more evals, each vastly cheaper. The net has **not** been measured on a real run yet. Compare against the diagnosed baseline (job `11809659`: ~5h20m of a 6h budget in round-evals) and record the result here either way.

**Until that measurement exists:** launch `--aug-type=grad_corr` jobs with generous `--time`, or raise `--round_interval` above the `max_iters // 20` default.

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
| `sensaug/hooks/grad_sens_analysis.py` | `PerturbationSensitivityAnalysisHookWithGradients` — builds the cross-correlation matrix R from the sweep, and publishes `red(a)` |
| `sensaug/redundancy.py` | `compute_red` / `reweight` — Lever 3's mechanism. Pure numpy, no mmseg |
| `scripts/calibrate_lambda.py` | offline λ sweep against logged `corr_matrix_log.json` files |
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
| `corr_redundancy_log.txt` | correlation pipeline | JSONL, one record per emission: per-op `red(a)` (standardized) and the raw row sums |

Launch tensorboard:
```bash
tensorboard --logdir experiments/ --host 0.0.0.0
```

## Nexus cluster

Reference config at `configs/nexus.yaml` — shows the full set of supported datasets (cityscapes, ade20k, pascal_voc12, loveda, potsdam, synapse, a2i2haze, acdc, idd). Della currently only has Cityscapes.
