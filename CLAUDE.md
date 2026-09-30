# sensaug — Sensitivity-Informed Augmentation for Robust Segmentation

Codebase contact: Laura Zheng <lyzheng@umd.edu>

## What this repo does

Trains segmentation models with adaptive, sensitivity-informed augmentation. The core idea: periodically run a perturbation sensitivity analysis (SA) on the model, then weight training augmentations toward the perturbation types the model is currently worst at.

Built on top of MMSegmentation + MMEngine. Custom components (hooks, loops, dataset classes, transforms) are registered via MMSeg/MMEngine registries.

## Cluster & environment

Two clusters, one config file each. **Nexus is where the current work runs** — this
checkout lives at `/gammascratch/amodak/SensAug` and every experiment under
`experiments/` was produced there.

| | **Nexus** (UMD) | **Della** (Princeton) |
|---|---|---|
| cluster config | `configs/nexus.yaml` | `configs/della.yaml` |
| launcher | `job_scripts/train_nexus_gamma.sbatch` | `job_scripts/train_della{,_1gpu,_4gpu}.sbatch` |
| partition / account | `gamma` (fallbacks: `scavenger`, `tron`) | — |
| GPUs | 4× `rtxa6000`, DDP via `torchrun` | 1 or 4 |
| datasets | all 9 in `configs/nexus.yaml` | cityscapes only |
| internet on compute nodes | **yes** | **no** — see below |

- Conda env: `sensaug` (Python 3.10, PyTorch 2.0, MMSeg latest). On Nexus,
  `module load miniforge3` first — `conda` is not on a fresh compute-node PATH.
- CUDA module: `cuda/11.7.0` on both.
- Set all paths in the cluster config, never in code.
- **Della's compute nodes have no internet access.** Backbones whose config
  downloads a pretrained checkpoint at `model.init_weights()` time (segformer,
  pspnet-rsb, convnext, swin) need that checkpoint pre-staged — see
  [Pretrained backbone checkpoints](#pretrained-backbone-checkpoints) below. Nexus
  needs none of that.

## Datasets

On **Nexus**, `configs/nexus.yaml` carries all nine (cityscapes, ade20k,
pascal_voc12, loveda, potsdam, synapse, a2i2haze, acdc, idd) and
`scripts/prepare_datasets.py` installs seven of them — see [Adding a new
dataset](#adding-a-new-dataset-3-steps-from-readme). Recent `grad_corr` and `ours`
runs are on pascal_voc12, loveda, acdc and ade20k.

On **Della** only Cityscapes is configured, and its setup is unfinished. Data lives
at:
```
data/cityscapes/
  gtFine/          ← extracted (annotations)
  leftImg8bit/     ← NOT YET EXTRACTED (see setup steps below)
  leftImg8bit_trainvaltest.zip
  gtFine_trainvaltest.zip
```

### Cityscapes setup on Della — pending steps

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

### Submit to Nexus (what the current experiments use)
```bash
# positional: aug-type backbone dataset [exp_name]
# defaults:   ours segformer pascal_voc12
sbatch job_scripts/train_nexus_gamma.sbatch grad_corr segformer pascal_voc12
```

Everything else is env vars, so a variant needs no edit to the script: `NAME`,
`RUN_TAG` (suffixes the exp name for repeat runs), `CONFIG`, `WORK_DIR`,
`RANDOM_PRUNE_COUNT` / `RANDOM_PRUNE_SEED` (the `null` arm; seed defaults to the
SLURM job id and is always passed explicitly so `EXP_NAME` carries it),
`CONDA_ENV`, `REPO_DIR`, `ALLOW_RESUME`, and for Lever 3 `DOWNWEIGHT` (default
`mRMR`) / `CORR_LAMBDA` (default `0.25`) / `PRUNED_AUGS`. It appends
`--corr-downweight-method` + `--corr-lambda` only on `grad_corr` runs, because
train.py warns they are inert elsewhere. Logs land in
`job_scripts/logs/<job-name>/`. It runs `torchrun` with one process per GPU and
always passes `--launcher=pytorch --no-inv-aug --auto-scale-lr --descending-MA`.

### Submit to Della
```bash
# defaults: aug=none, backbone=pspnet, dataset=cityscapes
sbatch job_scripts/train_della.sbatch

# override positional args
sbatch job_scripts/train_della.sbatch ours segformer cityscapes
```

### Run locally
```bash
python train.py \
  --cluster-config=configs/nexus.yaml \
  --backbone=segformer \
  --dataset=pascal_voc12 \
  --aug-type=none \
  --work_dir=./experiments \
  --exp_name=none_segformer_pascal_voc12 \
  --no-inv-aug
```

### Key train.py flags
| Flag | Values | Notes |
|---|---|---|
| `--aug-type` | `none`, `ours`, `default`, `grad_corr`, `random`, `autoaugment`, `augmix`, `randaugment`, `trivialaugment`, `idbh`, `vip` | `ours` = sensitivity-informed. `grad_corr` = **enables the gradient cross-correlation pipeline** (logs the matrix R) — it is its own `--aug-type` value, not a separate flag. (A standalone `--grad-corr` flag existed at one point; it's gone — `--aug-type=grad_corr` is the only way to turn the correlation pipeline on.) |
| `--backbone` | `pspnet`, `segformer`, `convnext`, `deeplabv3plus`, `swin`, `mae`, `vit` | must have a config under `sensaug/custom_configs/mmseg/<backbone>/` |
| `--dataset` | keys from the cluster config's `datasets:` | 9 on Nexus; only `cityscapes` on Della |
| `--no-inv-aug` | flag | exclude color/photometric augmentations |
| `--no-warmup` | flag | skip clean-training warmup rounds |
| `--no-corr-sa` | flag | under `--aug-type=grad_corr`, disable the SA loop — trains exactly like `none` while still running the correlation measurement. This is the control arm |
| `--resume` | flag | auto-resume from last checkpoint in work_dir |
| `--rounds-config` | path | the **val-round grid**: how many rounds a run has, how many are warmup, and which of them the correlation pipeline fires on. Default `configs/rounds.yaml`. Cluster-independent on purpose — see [The round schedule](#the-round-schedule) |
| `--round_interval` | int (iters) | the **SA pipeline's clock**. Default `max_iters // n_rounds` from `rounds.yaml` (20 → `max_iters // 20`). Overrides `schedule.round_interval` |
| `--corr-interval` | int (iters) | put the correlation pipeline on a **fixed clock of its own** instead of the round-aligned default. Only meaningful under `--aug-type=grad_corr`. Overrides `schedule.corr_interval`. No default any more — unset means the round schedule |
| `--corr-sync-sa` | flag | fire the correlation pipeline on **every** SA round rather than the round-aligned subset. ~5× the sweeps at the default geometry. Takes precedence over `--corr-interval` |
| `--corr-lambda` | float | redundancy down-weighting strength (**Lever 3**). `0` (default) leaves the pdf bit-identical — the control arm. Under `soft-weighting` it is the tilt strength; under `mRMR` it is the prune budget (`ceil(A/(1+λ))` ops survive). See below |
| `--corr-red-mode` | `squared`, `abs`, `signed` | how R reduces to a redundancy quantity. `squared` default. `soft-weighting` reduces a whole row to one score per op; `mRMR` applies the same reduction cell-by-cell and keeps it pairwise |
| `--corr-downweight-method` | `none`, `soft-weighting`, `mRMR` | which function turns R into the reweighted pdf. `none` = pdf used as generated (R still measured and logged, just not fed back); `soft-weighting` = the max-entropy tilt; `mRMR` = **hard pruning** — rank the ops by minimum-Redundancy Maximum-Relevance and zero everything outside the budget. **Required** on every `--aug-type=grad_corr` run (no default — an arm is never left unnamed); a WARNING and otherwise inert on every other arm, `--no-corr-sa` included. See [Adding a down-weighting method](#adding-a-down-weighting-method) |
| `--corr-lambda-ramp` | `linear`, `constant` | ramp λ from 0 over training (default) vs. full strength from the first R emission |
| `--corr-keep-within-op` | flag | keep the `lighter_X`/`darker_X` and `_pos`/`_neg` cells in `red(a)`; excluded by default |
| `--corr-skip-pruned-eval` | flag | ⚠️ **rejected on `--aug-type=grad_corr`** — the only arm where it could act. See [Static pruning](#static-pruning----pruned-augmentations) |
| `--pruned-augmentations` | op names | permanently exclude ops from this whole run. Overrides (does not merge with) the cluster config's `pruned_augmentations:`. See [Static pruning](#static-pruning----pruned-augmentations) |
| `--hold-none-prob` | flag | hold P(no augmentation) fixed as `--pruned-augmentations` shrinks the bank, so the prune changes *which* aug is sampled and not *how often* one is. Off by default. See [Static pruning](#static-pruning----pruned-augmentations) |
| `--random-prune-method` | `none`, `null` | the random-pruning arm. `null` draws a fixed random subset of ops and prunes them for the whole run — the **control for mRMR**. Default `none` (no draw). See [The `null` random-pruning arm](#the-null-random-pruning-arm) |
| `--random-prune-count` | int | how many ops `null` **drops** (not keeps). Required on that arm, no default |
| `--random-prune-seed` | int | optional. Omitted → derived from `SLURM_JOB_ID`, else `TORCHELASTIC_RUN_ID`, else (single-process only) `os.urandom`; a multi-rank launch with none of those is **refused**. Whatever is used is logged and written to `random_prune.json` |
| `--sa_interval` | int | ⚠️ dead flag — parsed but never read. SA-curve recompute is every `SA_CURVE_CADENCE`-th round (6), defined in `sensaug/round_schedule.py` and applied in `sensaug/loops/sensaug_loop.py` |
| `--geometric-only` / `--photometric-only` | flag | restrict the op bank to the 10 geometric or the 22 photometric ops. Referenced all over this file; applied at `GpuAugSegDataPreProcessor._op_bank` and `resolve_perturbation_set` |
| `--random-aug` | flag | sample the bank uniformly instead of from the SA pdf |
| `--uniform` | flag | build a flat pdf (the SA curve is still measured and logged, just not used to shape it) |
| `--weighted-augs` | flag | `generate_pdf_new_weighted_aug` instead of `generate_pdf_new` — ops are not treated equally |
| `--descending-MA` | flag | prioritize *less* severe augmentations in the pdf. The Nexus sbatch passes this on every run |
| `--corr-magnitude-mode` | `mode`, `sampled_shared`, `sampled_independent`, `fixed` | how each op's probe magnitude is drawn from its SA distribution. `mode` (default) is the modal level, constant across the batch; `fixed` always uses the 0.5 reference. Ignored without a snapshot |
| `--corr-magnitudes` | path | seed the probe's magnitude snapshot from a JSON file, for a run with no SA of its own (e.g. `sensaug/testing/shared_levels_diff32.json`). A live SA snapshot supersedes it |
| `--use-foundation-backbone` | flag | DINOv2 backbone |
| `--freeze-early-layers` | flag | freeze the backbone's early layers |
| `--amp` / `--adamw` / `--auto-scale-lr` | flag | mixed precision, AdamW, LR scaled to the real batch size |
| `--launcher` | `none`, `pytorch`, … | `pytorch` under `torchrun` (what both 4-GPU sbatches pass) |

## The two pipelines

Training runs two independent measurements. They share no hook point and no code path — do not couple them. They *are* scheduled onto the same round grid by default, which is a different thing: see [The round schedule](#the-round-schedule).

| Pipeline | Question it answers | Fires | Code |
|---|---|---|---|
| Sensitivity analysis (SA) | which perturbations is the model *worst at*? (weights the training aug PDF) | every `round_interval` | `sensaug/loops/sensaug_loop.py` → `RobustValLoop` |
| Gradient cross-correlation | which perturbations are *redundant with each other*? (the matrix R) | on the rounds `configs/rounds.yaml` names (shipped default 3, 4, 7, 10, 13, 16 of 20, via `corr_cadence: 3`); `--corr-interval` for a fixed clock instead | `sensaug/round_schedule.py` → `sensaug/hooks/grad_hook.py` → `sensaug/hooks/grad_sens_analysis.py` |

- **SA** runs under `--aug-type=ours` *and* `--aug-type=grad_corr` (the latter unless `--no-corr-sa`), from the val loop. Both resolve `cfg.val_cfg.perturbation_set` to `diff32`; every other arm that builds an SA loop gets `legacy20`. The SA *curve* is recomputed every 6th round (`SA_CURVE_CADENCE`), so its effective cadence is `6 × round_interval`.
- **Which val loop runs.** `--aug-type=ours` → `RobustValLoop`; `--aug-type=grad_corr` → `GradCorrValLoop`, which subclasses it and adds exactly one thing, Lever 3's redundancy reweighting. The base declares `_apply_redundancy_reweighting` as an identity function and the subclass overrides it, so the three pdf generators and `run()` carry no `if grad_corr` branch. `--aug-type=grad_corr --no-corr-sa` builds neither — mmengine's stock `ValLoop`.
- **Correlation** is opt-in via `--aug-type=grad_corr` — it is its own `--aug-type` value, not a flag layered on top of another one (`--aug-type=none --grad-corr` is not valid; that flag doesn't exist). It fires from `after_train_iter`, freezing the model and sweeping the whole clean val set (500 images on Cityscapes) for `d loss / d magnitude` per aug per image. `CollectGradientHook` (priority NORMAL) sweeps; `PerturbationSensitivityAnalysisHookWithGradients` (priority LOW) correlates the sweep it just wrote — the priority ordering is load-bearing. Both are given the same gate — an explicit `fire_iters` set by default, an `interval` under `--corr-interval` — normalized by `resolve_gate()` and applied by `fires_at()`, both in `grad_hook.py`. For the unaugmented control arm — R measured against a baseline with no training augmentation, which is what the SA-on number gets compared to — pass `--aug-type=grad_corr --no-corr-sa`: that disables the SA loop, so the run trains exactly like `none` while still running the correlation measurement.

## The round schedule

A **round** is one run of the val loop: every `round_interval` training iterations,
`RobustValLoop` re-evaluates perturbation robustness and rebuilds the training pdf. The
first `warmup_rounds` do nothing but train; the SA *curve* the pdf derives from is
recomputed every 6th round after that. The grid is configured in
[`configs/rounds.yaml`](configs/rounds.yaml), selected with `--rounds-config`:

```yaml
n_rounds: 20          # sets the DEFAULT round_interval (max_iters // n_rounds)
warmup_rounds: 4      # --no-warmup forces 0
corr_rounds: null     # null -> derived (below); or a literal list of round numbers
control_rounds: null  # null -> derived as the firing rounds inside warmup
corr_cadence: 3       # rounds between correlation emissions; null/6 -> SA-curve-only
```

It is deliberately **not** in `configs/della.yaml` / `configs/nexus.yaml`: the round grid
is an experiment parameter, and a copy per cluster is a copy that drifts. The cluster
configs keep only the two `schedule:` intervals. A run copies its `rounds.yaml` into the
work_dir next to `seg_config.yaml`.

**Where the correlation pipeline fires.** At the shipped defaults (`corr_cadence: 3`),
rounds **3, 4, 7, 10, 13, 16** of 20:

| rounds | what |
|---|---|
| 0–2 | nothing |
| 3 | one **control probe** — the baseline, does *not* feed the pdf |
| 4 | compute, **SA-curve recompute** — governs rounds 4–9 |
| 7 | compute — refreshes `red(a)` inside the still-current 4–9 window; the curve itself doesn't move here |
| 10 | compute, **SA-curve recompute** — governs rounds 10–15 |
| 13 | compute — refreshes `red(a)` inside the still-current 10–15 window |
| 16 | compute, **SA-curve recompute** — governs rounds 16–18 |
| 19 | nothing (training's over) |

Setting `corr_cadence: null` (or `6`, `SA_CURVE_CADENCE`) drops back to the sparser
`3, 4, 10, 16` table — one emission per SA-curve recompute only, and no in-between
refreshes.

Three things make the SA-curve rounds (4, 10, 16) the load-bearing ones, and all three
are what the old `corr_interval = max_iters // 4` clock got wrong:

- **4 / 10 / 16 are the SA-curve recompute rounds.** R exists to be read by the pdf, and
  the pdf only changes shape when the curve behind it does. Each of those emissions then
  stays current for exactly the 6 rounds that curve governs — `corr_cadence`'s extra
  rounds (7, 13 above) don't change that; they just publish a fresher `red(a)` from a
  new gradient sweep partway through an already-current window.
- **The R measured at round `r` is read by round `r`'s own pdf.** `IterBasedTrainLoop`
  calls `after_train_iter` and only *then* `val_loop.run()`, so the sweep at iteration
  `(r+1) × round_interval` lands one hook point before the val loop that consumes it.
  (It therefore probes at the *previous* round's magnitudes, which is the right
  semantics — those were in effect over the window being measured.)
- **Round 19 never fires.** Training ends with it, so nothing could read its R. Under the
  interval clock the final iteration always fires, which is where the fourth, unusable
  matrix of every old `grad_corr` run came from.

**The control probe (round 3)** is the last warmup round, so R there is measured on a
model no pdf has ever touched — the baseline every later matrix is read against.
`PerturbationSensitivityAnalysisHookWithGradients` records it in `corr_matrix_log.json`
with `"role": "control"` and publishes *nothing* to `runner.corr_redundancy` — not even
clearing it. Round 4's emission would overwrite it before any pdf read it anyway; the
withholding is explicit so that "the baseline never touched training" is a property of
the code rather than a coincidence of two hook orderings. Every other emission carries
`"role": "active"`.

The count is derived from the round grid and `corr_cadence`, never pinned: at the
shipped `corr_cadence: 3`, `--round_interval=2000` on an 80k run gives 40 rounds and 13
emissions (1 control + 12 active) instead of 6. The schedule always follows the round
grid the run *really* has (`max_iters // round_interval`), not `rounds.yaml`'s
`n_rounds`, which is only the default divisor.

**Escape hatches**, in precedence order: `--corr-sync-sa` fires on every round;
`--corr-interval N` (or `schedule.corr_interval`) restores a fixed clock independent of
the rounds. Short of those, `corr_cadence: N` in the YAML changes how densely
emissions land *without* unaligning them from the round grid — lower than the shipped
`3` for more (and proportionally more expensive) sweeps, `null`/`6` to fall back to
SA-curve-recompute-only. Setting `corr_rounds:` / `control_rounds:` in the YAML pins an
arbitrary schedule instead (and ignores `corr_cadence`, since there's nothing left to
derive) — validated against the grid, so a round that could never fire is an error and a
control round outside the firing set is an error. `sensaug/round_schedule.py` holds all of
it as pure integer math (no torch, no mmseg), tested in `tests/test_round_schedule.py`.


## Static pruning (`--pruned-augmentations`)

Removes named ops from a run entirely — the stage-2 half of
`job_scripts/train_mrmr_prune_retrain.sbatch`, where stage 1 measures R under
`--corr-downweight-method=mRMR` and stage 2 retrains with what it pruned. Distinct
from mRMR's own pruning in one way that drives everything below: **a statically
pruned op is gone for the whole run and cannot return**, where an mRMR-pruned op
is re-derived every round and may come back.

It reaches five places. The first two were always there; the rest were added
because "we removed this op" has to be true system-wide, not only in the sampler:

| where | what |
|---|---|
| `RobustValLoop.update_sa_curve` | `exclude=`, at the SA curve's source — so the op never reaches the round-eval, and never reaches any pdf built from the curve |
| `GpuAugSegDataPreProcessor._op_bank` | dropped from the uniform sampling bank (`--random-aug`) |
| `CollectGradientHook` | `static_pruned_ops`: no forward/backward, ever. **Not** routed through `skip_pruned` — that path backfills from `_last_full_row`, and a statically pruned op has no prior measurement to backfill from |
| `corr_matrix_log.json` | recorded as `"static_pruned"`, kept **out of** `"dropped"`. `dropped` means "measured, no variance" and stays a health signal worth chasing; `static_pruned` is a fact about the run's configuration |
| the pdf's op count | only under `--hold-none-prob` — see below |

R stays **32×32**: a pruned op keeps its row and column, filled with NaN, so
`corr_matrix_log.json` is comparable across arms and `corr_viz` shades those
rows distinctly (`STATIC_PRUNED_FILL`) rather than hatching them like a failed
measurement.

**`--hold-none-prob`, and why it is off by default.** The pdf carries a synthetic
`("none", 0)` entry whose mass is derived from the count of surviving ops, so
pruning 6 of 30 raises P(no augmentation) by ~0.7 pp. The pruned arm then trains
on clean images that much more often than its control — a second difference
inside a two-arm comparison, and a violation of the invariant `_downweight_mrmr`
already holds ("changes which augmentation is sampled, never how often"). With
the flag, the denominator counts the pruned ops as if still present and their
mass goes to the survivors. Off by default so a run launched before the flag
existed is reproduced exactly; **pass it on both arms or neither.**

Two things it does *not* cover, both pre-existing and both pinned in
`tests/test_static_prune.py`:

- `generate_pdf_new` evaluates `betabinom.pmf(i, len(levels), 0.75, 1.0)` for
  `i` one short of that pmf's support, so ~13% of the perturbation mass falls
  through to `("none", 0)` and the real P(none) is ~16%, not the `1/(N+1)` ≈ 3%
  the code reads as. `--hold-none-prob` holds the *allotment* fixed, so it holds
  the leaked fraction fixed too — the drift is fixed either way.
- `generate_pdf_new_weighted_aug` (`--weighted-augs`) drifts too, from that same
  truncation being a function of the op count (~6.4% at 10 ops, ~5.5% at 30).
  Same order, different mechanism, **not** addressed by this flag: correcting it
  means choosing how to renormalize a truncated pmf, which changes the shape of
  the ranking rather than its scale.

**`--corr-skip-pruned-eval` is rejected on `--aug-type=grad_corr`.** It would skip
the sweep for mRMR-pruned ops and backfill each from an *earlier* checkpoint's
gradients, which are then correlated against freshly measured rows — so one R
mixes vintages and successive emissions stop being comparable. Watching R change
as the model improves is what a grad_corr session is for, so a stage-1
measurement run sweeps everything on purpose. The flag stays inert-with-a-warning
on every other arm (nothing but `GradCorrValLoop` publishes
`runner.corr_pruned_ops`). To actually stop training on an op, prune it
statically.

### The `null` random-pruning arm

`--random-prune-method=null --random-prune-count N` answers the one question
mRMR invites: **does ranking ops by redundancy actually beat dropping the same
number of them at random?** Before training starts it draws N ops at random and
hands them to `--pruned-augmentations`, so from that point on it *is* a static
prune — same five call sites, same `"static_pruned"` bookkeeping, same NaN rows
in a 32×32 R. Nothing under `sensaug/` knows this arm exists.

It differs from mRMR exactly where a control should: the prune is **fixed for the
run and never re-derived**, carries no R, no λ, no correlation pipeline, and no
`GradCorrValLoop`. It is deliberately *not* a `DOWNWEIGHT_METHODS` entry — those
dispatch on a published R this arm never has.

- **N is the count DROPPED**, so it matches a finished stage-1 run mechanically:
  `wc -w < experiments/<stage1>/mrmr_pruned_ops.txt`.
- **The pool is narrower than the bank**, and that is what makes N mean N. It
  excludes `--no-inv-aug`'s two ops (which the Nexus sbatch passes on *every*
  run, so the pool is 30, not 32), the `--geometric-only` / `--photometric-only`
  filter, and anything already on `--pruned-augmentations` — the draw composes
  with an explicit list rather than overlapping it. Pruning the whole pool is
  refused: it would leave a pdf of nothing but `("none", 0)`.
- **The seed is never generated per rank.** Every rank runs `train.py`
  independently and torch.distributed is not initialized at argparse time, so
  there is no collective to sync with — and `mmengine.dist.sync_random_seed()`
  does not help, since with dist uninitialized it sees `world_size == 1` and
  returns a per-process seed without broadcasting. A generated seed would train a
  different bank on each GPU, averaged into one update, invisibly. Hence the
  precedence in the flags table above, and the outright refusal when nothing is
  derivable. `train_nexus_gamma.sbatch` always passes one explicitly (the job id
  by default) so `EXP_NAME` can carry it.
- **The draw uses a private `RandomState`**, never the global numpy RNG that
  `set_manual_seed` pins to 0 and `GpuAugSegDataPreProcessor` samples every
  per-image op from — consuming from that stream here would shift every later
  draw as a function of the pruned count, so the two arms would differ by more
  than their banks.
- **Both the count and the seed go in the exp name** (`_nullprune6_s17`).
  Two draws of the same size at different seeds are different experiments; without
  the seed they would share a work_dir and the resume guard would have them
  silently resume each other.
- **`{work_dir}/random_prune.json`** records seed, seed source, pool, dropped and
  kept. Neither the seed nor the draw is recoverable from a checkpoint, and the
  seed is not necessarily in the launch command either.
- **`--hold-none-prob` applies here exactly as it does to any other prune**: pass
  it on both arms or neither.

Launching a matched arm against a finished mRMR stage 1:

```bash
RANDOM_PRUNE_COUNT=$(wc -w < experiments/<stage1>/mrmr_pruned_ops.txt) \
  sbatch job_scripts/train_nexus_gamma.sbatch ours segformer ade20k
```

Legal on any arm with a non-empty pool (`ours`, `default`, `grad_corr`,
`random`); every other `--aug-type` samples from no registry and is refused.

**`--no-inv-aug` is now excluded at the same source.** `_remove_H_perturbations`
used to pop `lighter_H`/`darker_H` from `miou_record` *after* `test_perturbed_new`
had already run a full 5-level eval on each; those names now go into
`update_sa_curve`'s `exclude=` alongside the pruned ops. The method is kept —
it still guards the `load_sa_curve()` path, where the curve is read off disk
unfiltered — and is a no-op on the `update_sa_curve` path.

## The three augmentation vocabularies

Everything above turns on which set of op names is in play. There are three, and
two of them share all 32 keys while meaning different implementations under each.

| `perturbation_set` | ops | keys | implementation | used by |
|---|---|---|---|---|
| `legacy20` | 20 | PascalCase (`BrightnessTransform`, `NegativeRotate`) | CPU cv2/numpy transform classes | `--aug-type=random`, and the CPU SA path |
| `non-diff32` | 32 | snake_case (`lighter_R`, `rotate_neg`) | the CPU transform classes | CPU consumers that need R's names |
| `diff32` | 32 | snake_case, **identical to `non-diff32`** | GPU-batched torch ops, applied by `sensaug/dataset/gpu_augment.py` | `--aug-type=ours`, `default`, `grad_corr` |

`resolve_perturbation_set()` (`sensaug/dataset/augmentations.py`) is the single
dispatch point. An import-time assertion pins
`set(NON_DIFF32_OPS) == set(DIFF32_OPS)`.

**Why the 32-key sets exist.** R is indexed by the 32 `DIFF32_OPS` names. The
training pdf is indexed by whatever `perturbation_set` resolves to. Under `legacy20`
those two overlap on *nothing*, so no per-op quantity read off R could ever index
into the pdf — which is what makes Lever 3 impossible on that set. The 32-key sets
give R's names to the pdf.

**Why `diff32` is the one the GPU arms run.** The ops are batched GPU tensor
functions by design, and applying them per-image on CPU (which is what a pipeline
transform and the SA round-eval both do) is 40–150× slower — that is what made
`grad_corr` round-evals the dominant cost of a run. `gpu_augment` applies them
batched, after collation, as preprocessor state, so the SA round-eval no longer
rebuilds the val dataloader once per (op, level) pair. `non-diff32` is the same 32
names through the CPU classes, for consumers that genuinely need a transform class.

One thing `non-diff32` does **not** give you: **magnitude scales are not calibrated
against `diff32`.** Same op identity, different units — `blur` is the starkest (cv2's
kernel-size-derived implicit sigma vs. sigma directly). A per-op *score* transfers
between the two; a magnitude does not.

**The 10 geometric ops warp the label with the image**, on every path. The CPU
classes do it by construction (`Rotate._rotate_seg`, and `Rotate._rotate` rebuilds
the cv2 matrix from whichever array it is warping). On the `diff32` path both
consumers — `CollectGradientHook._grad_for_op` and
`GpuAugSegDataPreProcessor._apply_ops` — call `geometric_affine_matrix` +
`warp_image_and_label` *instead of* the raw op, which reproduces the op's own image
warp exactly while carrying the label along. `warp_image_and_label` re-expresses the
matrix in the label's own pixel frame (`_matrix_in_frame`), because mmseg val
pipelines `Resize` the image and then `LoadAnnotations` at the original size, so the
two rasters routinely differ (pascal_voc12, ade20k, acdc; not cityscapes or loveda,
whose val scales are no-ops). Anything new holding a geometric key and a label goes
through that pair, not through `DIFF32_OPS[name]`.

The `red(a)` figures this file used to quote for geometric vs. photometric ops were
measured before that was true and have been removed; **they need re-measuring** — see
the verification note in "Known issue" below.

Because `perturbation_set` selects between two registries with identical keys,
`_perturbation_transform_cfg()` (`sensaug/runner_utils.py`) takes it as an argument
and raises rather than unpacking `diff32`'s raw callable into a transform class.

## Lever 3: redundancy down-weighting of the training pdf

`q(a) ∝ pdf_old(a) · exp(−λ · red(a))` — the closed-form solution to minimising
`KL(q ‖ pdf_old)` subject to a budget on `Σ q(a)·red(a)`. `red(a)` is the
standardized row sum of R. Lives in `sensaug/redundancy.py` (pure numpy, no mmseg).

- **Handoff.** `PerturbationSensitivityAnalysisHookWithGradients.prune_augmentations`
  (which no longer prunes) publishes `runner.corr_redundancy` and appends to
  `corr_redundancy_log.txt`; `RobustValLoop._apply_redundancy_reweighting` reads
  whatever is current. The loop never has to agree with the hook, and before the
  first R emission the pdf is untouched. By default the hook's schedule is aligned
  to the rounds (see [The round schedule](#the-round-schedule)), so on a firing
  round "whatever is current" is the matrix measured one hook point earlier — but
  nothing in the loop assumes it, and `--corr-interval` unaligns them again without
  changing a line of it. The round-3 control probe publishes nothing, so the first
  score the loop can ever see comes from round 4.
- **λ is portable because `red(a)` is standardized.** Verified against all four
  logged `corr_matrix_log.json` files (10 checkpoints, both A=14 and A=32):
  λ=0.1 → 1.4–1.6×, **λ=0.25 → 2.3–3.1×**, λ=0.5 → 5.2–9.7×, λ=1.0 → 27–94×.
  Portability itself degrades past λ≈0.5. Recompute any time with
  `python scripts/calibrate_lambda.py` (offline, no GPU).
- **λ=0 is bit-identical**, not merely close — it short-circuits. That is what makes
  it usable as the control arm.
- **The functional form is pluggable, and must be named.** `DOWNWEIGHT_METHODS` in
  `sensaug/loops/grad_corr_loop.py` maps `--corr-downweight-method` to the function
  that turns the published score into the pdf. Three arms today: `none` (pdf used as
  generated), `soft-weighting` (the max-entropy tilt above, a thin adapter over
  `redundancy.reweight` so `scripts/calibrate_lambda.py` can still sweep λ offline
  against it), and `mRMR` (hard pruning — see below). Adding a fourth is a function
  plus a dict line — see
  [Adding a down-weighting method](#adding-a-down-weighting-method).
- **`none` and `--corr-lambda=0` reach the same place by different routes.** `none`
  is the *named* no-down-weighting arm — it ignores λ entirely and says so in the
  log every round. λ=0 is the *numeric* one, and short-circuits inside
  `redundancy.reweight` before any method runs. Either is a valid control; `none` is
  the one that shows up in the launch command.
- **`("none", 0)` is held fixed.** Only the perturbation mass is redistributed, so
  Lever 3 changes *which* augmentation is sampled, never *how often* augmentation
  happens.
- **Three guards, all of which log rather than degrade silently:** the score is
  withheld when the shared-factor loading exceeds `_LOADING_ALARM` (0.9) — R is then
  ranking which *images* are hard; when no cell survives the FDR gate; and when
  `red(a)` has near-zero variance.
- **`soft-weighting` cannot reach zero, `mRMR` is built to.** exp() is strictly
  positive, so on the tilt an op is pushed down but never deleted — which at the
  correlation sizes actually observed (mean |r| 0.11–0.22) is the strongest claim the
  measurement supports. `mRMR` makes the stronger claim deliberately; see below.

### `mRMR`: the hard-pruning arm

`--corr-downweight-method mRMR` ranks the ops by minimum-Redundancy
Maximum-Relevance and sets everything outside the budget to probability **exactly
zero**. Lives in `sensaug/loops/grad_corr_loop.py` (`_downweight_mrmr`), not in
`redundancy.py` — unlike the tilt it has no offline consumer.

- **The ranking is textbook greedy mRMR** and carries no λ. Seed with the most
  relevant op, then repeatedly take the op maximising `rel(a) − red(a | S)` against
  the already-selected set `S`. Both terms are standardized across ops, so the
  difference is dimensionless and neither can dominate by unit choice.
- **Relevance is the SA loop's own pdf**, summed over an op's magnitude levels —
  already the pipeline's statement of which perturbations the model is worst at, so
  it needs no second signal and inherits `--uniform` / `--weighted-augs`. When the pdf
  is flat (before the SA curve exists, or a whole `--uniform` run) the ranking
  degenerates gracefully into pure minimum-redundancy selection.
- **Redundancy is pairwise**, off the same cells `compute_red` sums into `red(a)`,
  under the same within-op mask, the same FDR gate and the same `--corr-red-mode`
  reduction. The two arms differ in what they *do* with R, not in what they think R
  says.
- **λ is the budget**: `ceil(A/(1+λ))` of the A measured ops survive. λ=0 keeps
  everything (continuous with the short-circuit, not discontinuous at it), 0.25 keeps
  ~80%, 0.5 ~67%, 1.0 half, 2.0 a third. It composes with `--corr-lambda-ramp` the
  same way, so early rounds prune little and the bank narrows as R earns it. Note
  `scripts/calibrate_lambda.py` calibrates the **tilt's** λ, not this one.
- **The prune is not latched.** It is re-derived from the current pdf and the current
  R every round, so an op pruned at one round can return at the next — the
  alternative commits the run to a decision made off the earliest, least trustworthy R.
- **An op with no usable row in R is exempt**, never pruned. A heavily FDR-gated R
  therefore prunes little, and logs that it did.
- **Read it next to a `soft-weighting` run at the same λ.** Deletion is a stronger
  claim than mean |r| 0.11–0.22 supports on its own.
- **The geometric ops' R is worth re-reading before you trust a prune.** It was
  contaminated by image-label misalignment until `_matrix_in_frame` landed, which made
  the geometric ops look *least* redundant — so mRMR preferentially kept them and
  pruned photometric ops. That is fixed, but no mRMR run has been re-measured against
  the corrected R yet: check which ops a stage-1 run actually prunes before letting
  stage 2 retrain on the result.

To support a pairwise method, `runner.corr_redundancy` now also carries `names`, `r`
(the matrix), `mask_within_op` and `survives` alongside `red`/`raw`/`dropped`. `r` and
`survives` are **excluded from `corr_redundancy_log.txt`** — both are already written
in full to `corr_matrix_log.json` / `corr_bootstrap_log.txt` under the same `iter`,
so join on that rather than writing an A×A block per line.

### Adding a down-weighting method

The point of `--corr-downweight-method` is that the list grows. Two steps:

**Step 1** — write the function in `sensaug/loops/grad_corr_loop.py`, next to the
others:

```python
def _downweight_my_method(pdf_dict: dict, published: dict, lam: float):
    """One line on what it does, and why it is a different modelling claim."""
    ...
    return ReweightResult(new_pdf, applied=True, reason=None, spread=...)
```

**Step 2** — add one line to `DOWNWEIGHT_METHODS` in the same file:

```python
DOWNWEIGHT_METHODS = {
    "none": _downweight_none,
    "soft-weighting": _downweight_soft_weighting,
    "mRMR": _downweight_mrmr,
    "my-method": _downweight_my_method,
}
```

That is all. `train.py` builds the flag's `choices` from these keys, the CLI gate and
`resolve_downweight_method`'s error message list them, and the parametrized contract
tests in `tests/test_downweight_methods.py` pick the new arm up automatically.

**What a method can rely on**, so a new one doesn't re-derive it:

| | |
|---|---|
| `pdf_dict` | `(op, level) -> prob`, summing to 1, straight from whichever of the three pdf generators ran |
| `published` | the **whole** `runner.corr_redundancy` record, not just its `red` field — `names`, `r`, `survives`, `mask_within_op`, `mode`, `raw`, `dropped`, `iter`, `checkpoint` are all there, so a method built on the structure of R (as `mRMR` is) needs no signature change |
| `lam` | already ramped by `--corr-lambda-ramp`; never 0 (the caller short-circuits), so no method re-implements either |
| return | `redundancy.ReweightResult` — the applied/reason/spread logging in `_apply_redundancy_reweighting` is written against it |

**Three rules that are not optional:**

- **Hold `("none", 0)` fixed.** Reweighting it would let the method change how *often*
  augmentation happens, which is a different intervention from changing *which*
  augmentation happens, and the two would be inseparable in the results.
- **Decline, never raise.** The call happens mid-training on every rank. A method that
  cannot act returns `applied=False` with a `reason`; the loop logs it. Raising kills
  the job at round 4.
- **Be soft, or say you aren't.** A method may not drive an entry to exactly 0 unless
  its name is in `HARD_PRUNING_METHODS` (same file). Deletion is a stronger claim than
  the observed correlation sizes support, so it costs a line someone has to write on
  purpose rather than happening by accident.

All three are enforced by the parametrized tests, so a method that breaks them fails
immediately rather than at analysis time.

### DDP correctness: R must be built on every rank, not just rank 0

`PerturbationSensitivityAnalysisHookWithGradients._emit()`
(`sensaug/hooks/grad_sens_analysis.py`, fixed in `f71c990`) used to gate
the entire R computation behind `is_main_process()`. Under multi-GPU DDP (e.g.
`job_scripts/train_della_4gpu.sbatch`), `RobustValLoop` rebuilds **each rank's own**
train dataloader from **that rank's own** `runner.corr_redundancy` — so publishing the
score on rank 0 alone left ranks 1..N-1 training on an un-reweighted pdf. DDP then
averages every rank's gradients into one update, so Lever 3's reweighting silently
reached only `1/world_size` of the actual signal, with nothing in the logs to flag it.

**Fix:** every rank now runs `_emit()`. The pre-`_emit` gather (`merge_rank_buffers`)
already leaves all ranks holding identical input, and `_emit` is deterministic (fixed
`bootstrap_seed`, pure numpy), so every rank independently arrives at the same R and
the same `red(a)`. The `is_main_process()` guards moved *inside* `_emit`, now scoped
only to the side effects that must not be duplicated: the `corr_log_path` /
`bootstrap_log_path` read-modify-writes and the TensorBoard logging.

## Resolved: `--aug-type=grad_corr` walltime, and where a run's time actually goes

**The history.** `experiments/grad_corr_grad_corr_2_pspnet_cityscapes_4gpu_gradcorr`
(job `11809659`, 6h) was killed by SLURM at the time limit having reached iter
28000/80000. Not a crash: **training ≈ 38 min (~10%)**, one correlation sweep ≈ 4 min,
**SA round-evals ≈ 5h20m (~88%)** — two individual round-evals took 1h49m and 3h11m.
The cause was that the SA round-eval rebuilt the val dataloader once per
`(perturbation, magnitude)` pair and ran the differentiable ops through a CPU
per-image pipeline transform, which is 40–150× slower than the batched GPU form they
are written for.

**What fixed it** (`cc1e403`): the 32 ops are no longer a pipeline transform at all.
`GpuAugSegDataPreProcessor` (`sensaug/dataset/gpu_augment.py`) applies them batched on
GPU after collation, so a round-eval sets *preprocessor state* (`set_eval_spec`)
instead of rebuilding a dataloader, and the training pdf likewise becomes state
(`set_train_spec`) rather than a rebuilt train loader. Op names are unchanged, so
`publish_corr_magnitudes` still fires and `--corr-magnitude-mode`'s adaptive probing
survives — which is what ruled out the other candidate fix, switching the SA loop to
the disjoint `legacy20` vocabulary.

**The measurement that closes this out** — `grad-corr-pascal_voc12-2`, segformer,
4×A6000, 20k iters, 20 rounds, 6 correlation sweeps:

| | before (job `11809659`) | after |
|---|---|---|
| completed | 28000/80000, killed at 6h | **20000/20000 in 1h53m** |
| worst single non-training span | 3h11m | **11m51s** |
| median non-training span | — | 2m47s |
| correlation sweep | ~4 min | ~4m20s (unchanged — it was never the problem) |

So the pathology is gone and runs finish. **Round-evals are still the majority of
wall time** (≈95 of those 113 min), which is inherent: 30–32 ops × 4 levels × a full
val pass per round. If a run is tight on walltime, raise `--round_interval` above the
`max_iters // 20` default rather than reaching for `--corr-interval`.

## Cluster config: `configs/nexus.yaml` / `configs/della.yaml`

Parsed by `sensaug/cluster_config.py`. Controls all paths, **the two pipelines' iteration intervals**, and the run-wide static prune list. The *round grid* is not here — it lives in `configs/rounds.yaml`, see [The round schedule](#the-round-schedule). The two files have the same shape and differ only in paths, the dataset list, and `pretrained_cache_dir` (Della only).

```yaml
data_root: /gammascratch/amodak/SensAug/data          # della: /projects/PUCHALLA/...
mmconfig_path: /gammascratch/amodak/SensAug/sensaug/custom_configs/mmseg
primary_metric: mIoU

# Della only -- its compute nodes have no internet, so init_cfg URLs are
# redirected here. Populate with scripts/download_pretrained_checkpoints.py.
pretrained_cache_dir: /projects/PUCHALLA/LLP2024/tumor/pretrained_checkpoints

schedule:                       # both in ITERATIONS; null → default
  round_interval: null          # SA pipeline's clock. null → max_iters // n_rounds
                                #   (n_rounds from configs/rounds.yaml, i.e. // 20)
  corr_interval: null           # null → the correlation pipeline uses the ROUND
                                #   SCHEDULE, not a fixed interval. Set a number
                                #   only to opt back out onto a clock of its own.

# Whole-run op exclusions, regardless of --aug-type. See Static pruning below.
# --pruned-augmentations OVERRIDES this list rather than merging with it.
pruned_augmentations: []

datasets:                       # key → subfolder under data_root
  cityscapes: cityscapes        # nexus also has ade20k, pascal_voc12, loveda,
                                # potsdam, synapse, a2i2haze, acdc, idd
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

**Step 3** — Add to the cluster config you run on (`configs/nexus.yaml` / `configs/della.yaml`):
```yaml
datasets:
  cityscapes: cityscapes
  DATASETNAME: path/relative/to/data_root
```

On Nexus, seven of the datasets in `configs/nexus.yaml` are installed rather than
hand-prepared: `scripts/prepare_datasets.py` resolves the target directory from the
same `DATA_ROOT_LOOKUP` `train.py` uses, runs the vendored converters under
`sensaug/custom_configs/dataset_converters/` (pinned to mmsegmentation v1.2.2), and is
idempotent. `--check` reports install status without downloading; `--all` covers every
key. `pascal_voc12` and `loveda` are public and fully scripted; `potsdam`, `synapse`,
`acdc`, `idd`, and `a2i2haze` need a manual, logged-in download first — running with
no `--src` prints the exact registration URL and steps.
See the [README](README.md#setting-up-supported-datasets) for the full per-dataset table, or
`sbatch job_scripts/prepare_datasets.sbatch` to run it as a job.

## Key files

| File | Purpose |
|---|---|
| `train.py` | Main training entrypoint |
| `test.py` | Robustness testing on OOD datasets |
| `configs/nexus.yaml` / `configs/della.yaml` | Cluster paths/datasets/intervals (edit these, not code) |
| `sensaug/cluster_config.py` | Parses the YAML config |
| `sensaug/dataset/datasets.py` | Custom dataset class registrations |
| `sensaug/dataset/augmentations.py` | Custom MMSeg transform registrations, the three op registries (`LEGACY20_OPS`, `NON_DIFF32_OPS`, and the `diff32` dispatch) and `resolve_perturbation_set` |
| `sensaug/dataset/differentiable_augmentations.py` | The base 14 autograd-compatible ops (`DIFFERENTIABLE_PERTURBATIONS`), all photometric |
| `sensaug/dataset/differentiable_augmentations_aa.py` | The 18 AutoAugment-family ops, `DIFF32_OPS` (14 + 18 — what R is indexed by), `GEOMETRIC_OP_KEYS`, and the label-safe pair `geometric_affine_matrix` / `warp_image_and_label` |
| `sensaug/dataset/gpu_augment.py` | `GpuAugSegDataPreProcessor` — applies the 32 ops batched on GPU after collation, and the `set_train_spec` / `set_eval_spec` / `clear_spec` / `suspended_augmentation` state API |
| `sensaug/hooks/sensitivity_hooks.py` | Legacy SA hooks (**not registered by train.py** — the SA loop does this now) |
| `sensaug/hooks/grad_hook.py` | `CollectGradientHook` — the frozen-frame per-image gradient sweep, and the shared `fires_at()` clock |
| `sensaug/hooks/grad_sens_analysis.py` | `PerturbationSensitivityAnalysisHookWithGradients` — builds the cross-correlation matrix R from the sweep, and publishes `red(a)` |
| `sensaug/redundancy.py` | `compute_red` / `reweight` — Lever 3's mechanism. Pure numpy, no mmseg |
| `sensaug/round_schedule.py` | the val-round grid and the correlation pipeline's firing schedule — parses `configs/rounds.yaml`. Pure integers, no torch/mmseg |
| `configs/rounds.yaml` | the round grid itself: `n_rounds`, `warmup_rounds`, and which rounds R is measured on |
| `scripts/calibrate_lambda.py` | offline λ sweep against logged `corr_matrix_log.json` files |
| `scripts/compute_grad_corr.py` | recompute R for an already-trained checkpoint without retraining — drives the grad_corr hooks off a loaded model + val dataloader; SLURM wrapper is `job_scripts/compute_grad_corr.sbatch` |
| `scripts/calibrate_kid_magnitudes.py` | pick a per-op probe magnitude via KID (Kernel Inception Distance) so every op carries a comparable distortion budget, for checkpoints with no SA-published magnitude to draw from |
| `scripts/run_grad_corr.sh` | reference invocation chaining the two scripts above against a finished experiment |
| `sensaug/loops/sensaug_loop.py` | `RobustValLoop` — **the SA pipeline**. `--aug-type=ours` runs this (and `grad_corr` via its subclass) |
| `sensaug/loops/grad_corr_loop.py` | `GradCorrValLoop` — `RobustValLoop` + Lever 3's redundancy down-weighting. `--aug-type=grad_corr` runs this |
| `sensaug/loops/train_loops.py` | `RobustIterBasedTrainLoop`, `SubsetTestLoop`, `SubsetValLoop` — loops a real training job runs |
| `sensaug/loops/test_loops.py` | `DebugAugLoop`, `VisualizeSampleLoop`, `KIDTestLoop` — development/inspection only, no metrics |
| `sensaug/custom_configs/mmseg/` | Per-backbone MMSeg config files |
| `job_scripts/train_nexus_gamma.sbatch` | **Nexus SLURM job script — the one in use** |
| `job_scripts/train_della.sbatch` | Della SLURM job script |
| `job_scripts/train_mrmr_prune_retrain.sbatch` | two-stage mRMR measure-then-retrain, the consumer of `--pruned-augmentations` |
| `scripts/prepare_datasets.py` | scripted install of seven of the Nexus datasets; `--check` reports status without downloading |
| `scripts/eval_all_augmentations.py` | per-op robustness eval of a finished checkpoint (`job_scripts/eval_all_augmentations.sbatch`) |

## Experiments output

Saved to `./experiments/<exp_name>/`. Contains checkpoints, tensorboard logs, the
resolved mmseg config, and copies of both config files the run was launched with
(`seg_config.yaml` and `rounds.yaml`) — so a later analysis need not trust that
either still says what it said months ago.

The Nexus sbatch derives `<exp_name>` as
`<aug_type>_<NAME>_<backbone>_<dataset>_4gpu[_<RUN_TAG>]` and refuses to start on a
work_dir that already holds checkpoints unless `ALLOW_RESUME=1` — so a repeat of the
same config needs a fresh `RUN_TAG` rather than silently continuing the earlier run's
checkpoint under a different config.

Per-pipeline logs:

| File | Written by | Contents |
|---|---|---|
| `sa_curve_log.txt` | SA pipeline | JSONL, one SA curve per recompute |
| `perturb_eval.txt` | SA pipeline | JSONL, per-round perturbed eval metrics |
| `aug_gradient_log.txt` | correlation pipeline | JSONL, one record per sweep batch — every per-image gradient, so R is recomputable offline without retraining |
| `corr_matrix_log.json` | correlation pipeline | one JSON array, one record per emission: raw + scale-normalized R, shared-image-factor loadings, `role` (`control` for the pre-pdf baseline probe, `active` otherwise), and three disjoint absence fields — `dropped` (measured, no variance), `stale` (measured earlier, reused now) and `static_pruned` (never measured; see [Static pruning](#static-pruning----pruned-augmentations)) |
| `corr_bootstrap_log.txt` | correlation pipeline | JSONL, per-cell bootstrap CIs and BH-FDR q-values |
| `corr_redundancy_log.txt` | correlation pipeline | JSONL, one record per emission: per-op `red(a)` (standardized) and the raw row sums |
| `random_prune.json` | `--random-prune-method=null` only | the arm's seed, its source, the eligible pool, and the dropped/kept split — neither the seed nor the draw is recoverable from a checkpoint |

Launch tensorboard:
```bash
tensorboard --logdir experiments/ --host 0.0.0.0
```
