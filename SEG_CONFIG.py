REPO_NAME = "mmsegmentation"

# DATA_ROOT = "data/"
DATA_ROOT = "/fs/nexus-projects/robustness_datasets/segmentation"
MMCONFIG_PATH = "/fs/nexus-scratch/lyzheng/sensaug/sensaug/custom_configs/mmseg"

PRIMARY_METRIC = "mIoU"

DATA_ROOT_LOOKUP = {
    "cityscapes": f"{DATA_ROOT}/cityscapes",
    "ade20k": f"{DATA_ROOT}/ade/ADEChallengeData2016",
    "pascal_voc12": f"{DATA_ROOT}/VOCdevkit/VOC2012",
    "loveda": f"{DATA_ROOT}/loveDA",
    "potsdam": f"{DATA_ROOT}/potsdam",
    "synapse": f"{DATA_ROOT}/synapse",
    "a2i2haze": f"{DATA_ROOT}/a2i2haze",
    "acdc": f"{DATA_ROOT}/acdc",
    "idd": f"{DATA_ROOT}/idd",
}

SUPPORTED_DATASETS = list(DATA_ROOT_LOOKUP.keys())
SUPPORTED_BACKBONES = [
    "pspnet",
    "segformer",
    "convnext",
    "deeplabv3plus",
    "swin",
    "mae",
    "vit",
]
