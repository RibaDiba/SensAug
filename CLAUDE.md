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

## Cluster config: `configs/della.yaml`

Parsed by `sensaug/cluster_config.py`. Controls all paths.

```yaml
data_root: /projects/PUCHALLA/LLP2024/tumor/data
mmconfig_path: /projects/PUCHALLA/LLP2024/tumor/sensaug/custom_configs/mmseg
primary_metric: mIoU
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
| `sensaug/hooks.py` | Sensitivity analysis hooks |
| `sensaug/loops.py` | Custom train/val loops (RobustValLoop, etc.) |
| `sensaug/custom_configs/mmseg/` | Per-backbone MMSeg config files |
| `job_scripts/train_della.sbatch` | Della SLURM job script |

## Experiments output

Saved to `./experiments/<exp_name>/`. Contains checkpoints, tensorboard logs, and a copy of the cluster config used.

Launch tensorboard:
```bash
tensorboard --logdir experiments/ --host 0.0.0.0
```

## Nexus cluster

Reference config at `configs/nexus.yaml` — shows the full set of supported datasets (cityscapes, ade20k, pascal_voc12, loveda, potsdam, synapse, a2i2haze, acdc, idd). Della currently only has Cityscapes.
