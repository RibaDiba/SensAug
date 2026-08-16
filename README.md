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

Each dataset has different setup instructions. You can find the setup instructions for most of them (all but a2i2haze) on this [MMSeg tutorial link](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html).

Additionally, you can also contact me if you would like a zip file of the post-processed data for convenience: ```lyzheng@umd.edu```.

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
| `--round_interval` | int | `max_iters // 20` | Iterations between robustness re-evaluations — the **SA pipeline's clock**. Overrides `schedule.round_interval` in the cluster config. See [Scheduling](#scheduling-two-independent-pipelines) |
| `--no-corr-sa` | flag | False | Under `--aug-type=grad_corr`, disable the SA loop — trains exactly like `none` while still running the correlation measurement. This is the control arm |
| `--corr-interval` | int | `max_iters // 4` | Iterations between gradient sweeps and R emissions — the **correlation pipeline's clock**. Only meaningful under `--aug-type=grad_corr`. Overrides `schedule.corr_interval` in the cluster config |
| `--sa_interval` | int | None | ⚠️ Currently unused — parsed but never read. The SA-curve recompute cadence is hardcoded to every 6th round in `sensaug/loops/sensaug_loop.py` |
| `--adamw` | flag | False | Use AdamW optimizer instead of default SGD |
| `--amp` | flag | False | Enable automatic mixed-precision (AMP) training |
| `--auto-scale-lr` | flag | False | Auto-scale learning rate based on batch size |
| `--resume` | flag | False | Auto-resume from latest checkpoint in `work_dir` |
| `--launcher` | str | `none` | Job launcher: `none`, `pytorch`, `slurm`, `mpi` |
| `--local_rank` | int | 0 | Local rank for distributed training |

### Scheduling: two independent pipelines

Training runs **two separate measurement pipelines**, on two clocks that have nothing to do with each other. Both are set in iterations, in the `schedule:` block of the cluster config, and both can be overridden on the command line.

| Pipeline | What it measures | Clock | Where it lives |
|---|---|---|---|
| **Sensitivity analysis (SA)** | Which perturbations the model is currently *worst at* — used to weight the training augmentation PDF | `schedule.round_interval` / `--round_interval` (default `max_iters // 20`) | `sensaug/loops/sensaug_loop.py` (`RobustValLoop`) |
| **Gradient cross-correlation** | Which perturbations are *redundant with each other* — the correlation matrix R | `schedule.corr_interval` / `--corr-interval` (default `max_iters // 4`) | `sensaug/hooks/grad_hook.py` → `sensaug/hooks/grad_sens_analysis.py` |

```yaml
# configs/della.yaml
schedule:
  round_interval: 4000    # SA pipeline: a val/SA round every 4000 iters
  corr_interval: 20000    # correlation pipeline: a gradient sweep + R every 20000 iters
```

Leave a value `null` (or omit the `schedule:` block entirely) to take the default. Precedence is **CLI flag > cluster config > default**.

Notes on each:

- **SA pipeline.** Runs only under `--aug-type=ours`. Every `round_interval` iterations it re-evaluates perturbation robustness and rebuilds the training sampling PDF. The SA *curve* itself is recomputed every 6th round (hardcoded in `sensaug/loops/sensaug_loop.py`), so the effective SA-curve cadence is `6 × round_interval`.
- **Correlation pipeline.** Opt-in via `--aug-type=grad_corr` — it is its own `--aug-type` value, not a flag you layer on top of another one (there is no standalone `--grad-corr` flag; `--aug-type=none --grad-corr` is not valid). Every `corr_interval` iterations it freezes the model, sweeps the whole clean val set (500 images on Cityscapes) for `d loss / d magnitude` per augmentation per image, and correlates that sweep into R. It fires from `after_train_iter`, so it never depends on a val round happening. The final training iteration always fires, so the converged model's R exists even when `max_iters` is not a multiple of `corr_interval`.

R is a claim about the augmentation operators themselves, not about the `ours` training loop, so the pipeline also needs an unaugmented control arm to compare against. That's `--no-corr-sa`: it disables the SA loop, so the run trains exactly like `none` while still running the correlation measurement.

```bash
python train.py \
  --cluster-config=configs/della.yaml \
  --backbone=pspnet --dataset=cityscapes \
  --aug-type=grad_corr --no-corr-sa --corr-interval=20000 \
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
