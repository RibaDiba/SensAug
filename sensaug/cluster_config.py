import yaml


def load_seg_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    data_root = raw["data_root"]
    return dict(
        MMCONFIG_PATH=raw["mmconfig_path"],
        PRIMARY_METRIC=raw.get("primary_metric", "mIoU"),
        DATA_ROOT_LOOKUP={k: f"{data_root}/{v}" for k, v in raw["datasets"].items()},
        SUPPORTED_DATASETS=list(raw["datasets"].keys()),
        SUPPORTED_BACKBONES=raw["supported_backbones"],
    )
