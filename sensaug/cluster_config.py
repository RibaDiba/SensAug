import yaml


def load_seg_config(path):
    """
    Load segmentation configuration from a YAML file.
    
    Parameters:
        path: Path to the YAML configuration file.
    
    Returns:
        A dictionary containing model configuration, dataset paths, supported
        datasets and backbones, scheduling options, and the optional pretrained
        checkpoint cache directory.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    data_root = raw["data_root"]
    return dict(
        MMCONFIG_PATH=raw["mmconfig_path"],
        PRIMARY_METRIC=raw.get("primary_metric", "mIoU"),
        DATA_ROOT_LOOKUP={k: f"{data_root}/{v}" for k, v in raw["datasets"].items()},
        SUPPORTED_DATASETS=list(raw["datasets"].keys()),
        SUPPORTED_BACKBONES=raw["supported_backbones"],
        # Optional, and `schedule:` with every value left null parses to None
        # rather than {} -- so `or {}` rather than a plain .get default.
        SCHEDULE=raw.get("schedule") or {},
        # Optional. None means "don't redirect pretrained-checkpoint URLs" -- the
        # cluster is assumed to have node-level internet access (e.g. nexus.yaml).
        PRETRAINED_CACHE_DIR=raw.get("pretrained_cache_dir"),
    )
