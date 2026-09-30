# Sensitivity-Informed Augmentation for Robust Segmentation 

If you have any questions, please contact Laura Zheng at ```lyzheng@umd.edu```. Thank you!

## Getting Started 

### Environment Setup (Conda) 

First, create a conda environment with Python version 3.10:

```conda create --name bp-38 python=3.10```

Then, install the library dependencies via pip: 

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In case dependencies need to be installed manually, we use the following versions: 

- PyTorch 2.0.0 
- Latest version of MMSegmentation

## Setting the Right Paths 
This repo contains references to many local paths for purposes of config, datasets, and output directories. It is important to set these paths up prior to running experiments.

Here are the following files which need to be modified to your own system: 

- The cluster config for your cluster: [configs/nexus.yaml](configs/nexus.yaml) (UMD Nexus) or [configs/della.yaml](configs/della.yaml) (Princeton Della)
- [All convenience scripts in job_scripts folder](job_scripts)

The code reads the paths from these files throughout training and testing. Every entry point takes `--cluster-config configs/<cluster>.yaml`; the file is parsed by [`sensaug/cluster_config.py`](sensaug/cluster_config.py) and copied to `{work_dir}/seg_config.yaml` at the start of each run for reproducibility.

> The cluster YAMLs replace the old `SEG_CONFIG.py`, which no longer exists.

### Pretrained backbone checkpoints

Compute nodes on Della have no internet access. Several backbone configs (segformer,
pspnet-rsb, convnext, swin) point their `backbone.init_cfg` at a
`download.openmmlab.com` URL, which crashes `model.init_weights()` with a DNS error
if the job hits it on a compute node.

`train.py` redirects those URLs to `pretrained_cache_dir` (set in the cluster YAML)
and fails fast with a clear message — not the checkpoint-loading traceback — if the
file isn't cached there yet. Populate the cache once, from somewhere with real
internet access (a Della **login** node, or your own machine + `rsync`/`scp` — not
a compute node or an sbatch job):

```bash
python scripts/download_pretrained_checkpoints.py --backbone segformer
# or: --backbone pspnet convnext swin, or --all for every supported backbone
```

Use `--dry-run` first to see what would be fetched (and its size) without
downloading anything.

## Setting up Supported Datasets 

Currently, this repo supports all backbones provided by MMSegmentation and additionally the datasets listed under `datasets:` in the cluster config (see [configs/nexus.yaml](configs/nexus.yaml) for the full set):

Training datasets:
- Cityscapes
- ADE20K 
- PASCAL VOC 2012
- LoveDA
- POTSDAM
- Synapse
- A2I2Haze

Testing datasets: 
- ACDC
- Dark Zurich 
- Nighttime Driving 
- IDD

Cityscapes and ADE20K are already staged on Nexus. The other seven are installed
with **[`scripts/prepare_datasets.py`](scripts/prepare_datasets.py)**, which resolves
each target directory from the same `DATA_ROOT_LOOKUP` that `train.py` uses, runs the
vendored MMSeg converters ([`sensaug/custom_configs/dataset_converters/`](sensaug/custom_configs/dataset_converters), pinned to mmsegmentation v1.2.2), verifies file counts, and is idempotent (an already-installed dataset is skipped).

```bash
# see what is / isn't installed, download nothing:
python scripts/prepare_datasets.py --all --cluster-config configs/nexus.yaml --check
```

**Public — no login, fully scripted** (also available as `sbatch job_scripts/prepare_datasets.sbatch`):

```bash
python scripts/prepare_datasets.py pascal_voc12 loveda --cluster-config configs/nexus.yaml
```

| key | source | notes |
|---|---|---|
| `pascal_voc12` | `VOCtrainval_11-May-2012.tar` (Oxford VGG, with pjreddie mirror) | plain VOC2012; the SBD `aug` split is not needed for the shipped config |
| `loveda` | Zenodo record `5706578` (`Train/Val/Test.zip`, ~9.5 GB) | converted to `img_dir/`+`ann_dir/` |

**Gated — needs a one-time manual, logged-in download.** Run the command with no
`--src` to print the exact URL and steps; then re-run pointing `--src` at the
archive(s):

```bash
python scripts/prepare_datasets.py acdc --cluster-config configs/nexus.yaml --src /path/to/acdc_download
```

| key | where to register | archive(s) | extra step |
|---|---|---|---|
| `potsdam` | isprs.org UrbanSemLab benchmark | `2_Ortho_RGB.zip`, `5_Labels_all.zip` | tiled to 512×512 by the converter |
| `synapse` | synapse.org project `syn3193805` | `RawData.zip` (BTCV Abdomen) | `pip install nibabel`; train/val split written automatically |
| `acdc` | acdc.vision.ee.ethz.ch | `rgb_anon_trainvaltest.zip`, `gt_trainval.zip` | test-only; converter maps raw `val/` → `test/` |
| `idd` | idd.insaan.iiit.ac.in | `idd-segmentation.tar.gz` | test-only; run AutoNUE `createLabels.py` (`--id-type level3Id`) to make `*_gtFine_labelTrainIds.png` first |
| `a2i2haze` | no public source | post-processed zip from `lyzheng@umd.edu` | expects `imgs/{train,val}` + `labels/{train,val}` |

The MMSeg dataset-prepare tutorial (<https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html>) is the upstream reference for the converter behaviour.

## Training a Model 
To train a model, you can either call the Python training file [```train.py```](train.py) directly or use one of the convenience bash scripts provided in ```job_scripts```. 

There are many command-line arguments in the train.py script, which you can list with ```python train.py --help```. The convenience scripts like [```job_scripts/train_generic.sh```](job_scripts/train_generic.sh) help make training simple and reproducible.

### All `train.py` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cluster-config` | str | *(required)* | Path to YAML cluster config (e.g. `configs/della.yaml`) |
| `--work_dir` | str | *(required)* | Root directory where experiment output folders are saved |
| `--exp_name` | str | `ours_{backbone}_{dataset}` | Experiment name; creates a subfolder under `work_dir` |
| `--aug-type` | str | `none` | Augmentation strategy: `none`, `ours`, `default`, `grad_corr`, `random`, `autoaugment`, `augmix`, `randaugment`, `trivialaugment`, `idbh`, `vip`. `grad_corr` enables the gradient cross-correlation pipeline (see [Scheduling](#scheduling-two-independent-pipelines)) — it is its own value, not a separate flag |
| `--backbone` | str | `pspnet` | Model backbone (must exist under `sensaug/custom_configs/mmseg/`) |
| `--dataset` | str | `cityscapes` | Dataset key from cluster config |
| `--use-foundation-backbone` | flag | False | Use DINOv2 foundation model as backbone |
| `--geometric-only` | flag | False | Restrict augmentations to geometric transforms only |
| `--photometric-only` | flag | False | Restrict augmentations to photometric transforms only |
| `--no-inv-aug` | flag | False | Exclude color/photometric augmentations |
| `--no-warmup` | flag | False | Skip clean-training warmup rounds |
| `--random-aug` | flag | False | Sample augmentations randomly (instead of sensitivity-weighted) |
| `--weighted-augs` | flag | False | Weight augmentations unequally during sampling |
| `--uniform` | flag | False | Use uniform augmentation distribution |
| `--descending-MA` | flag | False | Prioritize less severe augmentations (descending moving average) |
| `--freeze-early-layers` | flag | False | Freeze early backbone layers during training |
| `--rounds-config` | path | `configs/rounds.yaml` | The **val-round grid**: how many rounds a run has, how many are warmup, and which of them the correlation pipeline fires on. See [Scheduling](#scheduling-two-independent-pipelines) |
| `--round_interval` | int | `max_iters // n_rounds` | Iterations between robustness re-evaluations — the **SA pipeline's clock**. Overrides `schedule.round_interval` in the cluster config |
| `--no-corr-sa` | flag | False | Under `--aug-type=grad_corr`, disable the SA loop — trains exactly like `none` while still running the correlation measurement. This is the control arm |
| `--corr-interval` | int | *(unset)* | Put the correlation pipeline on a fixed iteration clock instead of the round-aligned default. Only meaningful under `--aug-type=grad_corr`. Overrides `schedule.corr_interval` in the cluster config |
| `--corr-sync-sa` | flag | False | Fire the correlation pipeline on **every** SA round rather than the round-aligned subset. Takes precedence over `--corr-interval` |
| `--sa_interval` | int | None | ⚠️ Currently unused — parsed but never read. The SA-curve recompute cadence is every 6th round (`SA_CURVE_CADENCE` in `sensaug/round_schedule.py`) |
| `--adamw` | flag | False | Use AdamW optimizer instead of default SGD |
| `--amp` | flag | False | Enable automatic mixed-precision (AMP) training |
| `--auto-scale-lr` | flag | False | Auto-scale learning rate based on batch size |
| `--resume` | flag | False | Auto-resume from latest checkpoint in `work_dir` |
| `--launcher` | str | `none` | Job launcher: `none`, `pytorch`, `slurm`, `mpi` |
| `--local_rank` | int | 0 | Local rank for distributed training |

### Scheduling: two independent pipelines

Training runs **two separate measurement pipelines**. The SA pipeline runs on a fixed iteration interval; the correlation pipeline, by default, runs on a chosen subset of the *rounds* that interval produces — see [The round schedule](#the-round-schedule) below.

| Pipeline | What it measures | When it fires | Where it lives |
|---|---|---|---|
| **Sensitivity analysis (SA)** | Which perturbations the model is currently *worst at* — used to weight the training augmentation PDF | every `schedule.round_interval` / `--round_interval` iterations (default `max_iters // n_rounds`) | `sensaug/loops/sensaug_loop.py` (`RobustValLoop`) |
| **Gradient cross-correlation** | Which perturbations are *redundant with each other* — the correlation matrix R | on the rounds named by `configs/rounds.yaml` (default 3, 4, 10, 16 of 20), or every `--corr-interval` iterations if you name one | `sensaug/round_schedule.py` → `sensaug/hooks/grad_hook.py` → `sensaug/hooks/grad_sens_analysis.py` |

```yaml
# configs/della.yaml -- iteration intervals only
schedule:
  round_interval: 4000    # SA pipeline: a val/SA round every 4000 iters
  corr_interval: null     # null -> the correlation pipeline uses the round schedule
```

Leave a value `null` (or omit the `schedule:` block entirely) to take the default. Precedence is **CLI flag > cluster config > default**.

Notes on each:

- **SA pipeline.** Runs only under `--aug-type=ours`. Every `round_interval` iterations it re-evaluates perturbation robustness and rebuilds the training sampling PDF. The SA *curve* itself is recomputed every 6th round, so the effective SA-curve cadence is `6 × round_interval`.
- **Correlation pipeline.** Opt-in via `--aug-type=grad_corr` — it is its own `--aug-type` value, not a flag you layer on top of another one (there is no standalone `--grad-corr` flag; `--aug-type=none --grad-corr` is not valid). At each firing iteration it freezes the model, sweeps the whole clean val set (500 images on Cityscapes) for `d loss / d magnitude` per augmentation per image, and correlates that sweep into R. It fires from `after_train_iter` and calls nothing in the val loop, so it never *depends* on a val round happening — it is only scheduled alongside them.

#### The round schedule

A **round** is one run of the val loop. The grid lives in [`configs/rounds.yaml`](configs/rounds.yaml) — cluster-independent, because how many rounds a run has is an experiment parameter and not a path:

```yaml
n_rounds: 20          # sets the DEFAULT round_interval (max_iters // n_rounds)
warmup_rounds: 4      # --no-warmup forces 0
corr_rounds: null     # null -> derived; or a literal list of round numbers
control_rounds: null  # null -> derived as the firing rounds inside warmup
```

At those defaults the correlation pipeline fires on rounds **3, 4, 10, 16** of 20:

| rounds | what |
|---|---|
| 0–2 | nothing |
| 3 | one **control probe** — the baseline, does *not* feed the PDF |
| 4 | compute, used for rounds 4–9 |
| 10 | compute, used for rounds 10–15 |
| 16 | compute, used for rounds 16–18 |
| 19 | nothing (training's over) |

Rounds 4 / 10 / 16 are exactly the SA-curve recompute rounds, so each matrix stays current for the rounds that curve governs. The sweep runs from `after_train_iter`, which `IterBasedTrainLoop` reaches *before* it calls `val_loop.run()`, so the R measured at round `r` is on the runner in time for round `r`'s own PDF. Round 19 never fires: training ends with it, so nothing could read its R.

Round 3's probe is the last warmup round — R measured on a model no PDF has touched. It is logged like any other emission, with `"role": "control"` in `corr_matrix_log.json`, and is never published to the training PDF.

The emission count is derived, not pinned: `--round_interval=2000` on an 80k run gives 40 rounds and 7 emissions.

R is a claim about the augmentation operators themselves, not about the `ours` training loop, so the pipeline also needs an unaugmented control arm to compare against. That's `--no-corr-sa`: it disables the SA loop, so the run trains exactly like `none` while still running the correlation measurement.

```bash
python train.py \
  --cluster-config=configs/della.yaml \
  --backbone=pspnet --dataset=cityscapes \
  --aug-type=grad_corr --no-corr-sa \
  --work_dir=./experiments --exp_name=corr_baseline_pspnet_cityscapes
```

The correlation pipeline writes three files into `{work_dir}`:

| File | Contents |
|---|---|
| `aug_gradient_log.txt` | JSONL, one record per sweep batch — every per-image gradient, so R can be recomputed offline without retraining |
| `corr_matrix_log.json` | A JSON array, one record per emission: the raw and scale-normalized R, the ops dropped for zero variance, and the shared-image-factor loadings |
| `corr_bootstrap_log.txt` | JSONL, per-cell bootstrap confidence intervals and BH-FDR corrected q-values |

To train with the convenience script, simply run 

```bash job_scripts/train_generic.sh [aug] [model] [dataset]```

or, if you want to submit to a GPU cluster with a SLURM scheduler, you can simply run the same but with sbatch:

```sbatch job_scripts/train_generic.sh [aug] [model] [dataset]```

[aug] options: 'none', 'ours', 'default', 'grad_corr', 'random', 'autoaugment', 'augmix', 'randaugment', 'trivialaugment', 'idbh', 'vip'

[model] options: any model name from subfolders of [```custom_configs/mmseg```](sensaug/custom_configs/mmseg). example: 'pspnet', 'segformer', 'vit', 'swin'. 

[dataset] options: any key under `datasets:` in your cluster config. Each value is a path relative to `data_root`:

```yaml
# configs/nexus.yaml
data_root: /fs/nexus-projects/robustness_datasets/segmentation
datasets:
  cityscapes: cityscapes
  ade20k: ade/ADEChallengeData2016
  pascal_voc12: VOCdevkit/VOC2012
  loveda: loveDA
  potsdam: potsdam
  synapse: synapse
  a2i2haze: a2i2haze
  acdc: acdc
  idd: idd
```

Della currently only has Cityscapes set up — see [configs/della.yaml](configs/della.yaml).

NOTE: Our repo supports Tensorboard! You can launch Tensorboard while a model is training like so:
 
```tensorboard --logdir [work dir here] --host 0.0.0.0```

## Implementing a New Dataset 

If you would like to implement your own custom dataset, there are a few steps involved. 

### Step 1: Implement a new dataset MMSeg-style
Implement the dataset class in [```sensaug/dataset/datasets.py```](sensaug/dataset/datasets.py).
The existing custom datasets are short implementations because they subclass Cityscapes. For a more sophisticated implementation, you can check the official MMSeg tutorial: https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html 

### Step 2: Create a training config for the dataset 
MMSegmentation uses separate training configs for each dataset; it makes things easier for fine-tuning and whatnot. 

Our training script is set up to adapt any existing config for a dataset to any supported model, so only one new config is needed to support all models. 

In the past, we just create a new training config for the dataset in the path: [```custom_configs/mmseg/pspnet```](sensaug/custom_configs/mmseg/pspnet). This is purely because PSPNet already had many dataset implemented, and it is a light(er) model to test.

The [PSPNet config for A2I2Haze](sensaug/custom_configs/mmseg/pspnet/pspnet_r18-d8_4xb2-80k_a2i2haze.py) is entirely custom, so it may be easiest to make a copy of that file and swap out paths. 

Make sure the config file follows the same naming convention as all other configs, even if the naming convention is obscure. If you decide to make a copy of the A2I2Haze config, you can name the file like so: 
```pspnet_r18-d8_4xb2-80k_DATASETNAME.py``` 
Make note of the dataset name for the next step. 

### Step 3: Modify the cluster config 

Remember those cluster config files we keep referencing? It's time to modify them now: [configs/nexus.yaml](configs/nexus.yaml) and [configs/della.yaml](configs/della.yaml).

Add the new dataset under `datasets:`. The **key** is the name you will pass to `--dataset`, and must match the ```DATASETNAME``` you chose in the last step in the config naming. The **value** is the dataset's path relative to `data_root`:

```yaml
datasets:
  cityscapes: cityscapes
  DATASETNAME: path/relative/to/data_root
```

The training script pulls from this automatically ([`sensaug/cluster_config.py`](sensaug/cluster_config.py) resolves the full path as `{data_root}/{value}`). If all steps go smoothly, then you should be able to run the convenience script with the new dataset, with the ```DATASETNAME``` from the config you chose as the dataset argument. 
