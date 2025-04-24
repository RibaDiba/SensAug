REPO_NAME = "mmdetection"

DATA_ROOT = "/fs/nexus-projects/robustness_datasets/detection/data"
MMCONFIG_PATH = "custom_configs/mmdet"
# MMCONFIG_PATH = 'sensaug/mmdetection/configs/_base_'

PRIMARY_METRIC = "mAP"

DATA_ROOT_LOOKUP = {
    "voc0712": f"{DATA_ROOT}/VOCdevkit/",
}

SUPPORTED_DATASETS = list(DATA_ROOT_LOOKUP.keys())
SUPPORTED_BACKBONES = ["faster-rcnn_r50_fpn"]
