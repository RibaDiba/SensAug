import os
import sys
import tempfile
import textwrap
import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensaug.cluster_config import load_seg_config


SAMPLE_YAML = textwrap.dedent("""\
    data_root: /data/seg
    mmconfig_path: /repo/sensaug/custom_configs/mmseg
    primary_metric: mIoU
    datasets:
      cityscapes: cityscapes
      acdc: acdc
      idd: idd
    supported_backbones:
      - pspnet
      - segformer
""")


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text(SAMPLE_YAML)
    return str(p)


def test_load_seg_config_keys(config_file):
    result = load_seg_config(config_file)
    assert set(result.keys()) == {
        "MMCONFIG_PATH", "PRIMARY_METRIC", "DATA_ROOT_LOOKUP",
        "SUPPORTED_DATASETS", "SUPPORTED_BACKBONES", "SCHEDULE",
    }


def test_mmconfig_path(config_file):
    result = load_seg_config(config_file)
    assert result["MMCONFIG_PATH"] == "/repo/sensaug/custom_configs/mmseg"


def test_primary_metric(config_file):
    result = load_seg_config(config_file)
    assert result["PRIMARY_METRIC"] == "mIoU"


def test_data_root_lookup_constructed_correctly(config_file):
    result = load_seg_config(config_file)
    lookup = result["DATA_ROOT_LOOKUP"]
    assert lookup["cityscapes"] == "/data/seg/cityscapes"
    assert lookup["acdc"] == "/data/seg/acdc"
    assert lookup["idd"] == "/data/seg/idd"


def test_supported_datasets_matches_yaml_keys(config_file):
    result = load_seg_config(config_file)
    assert result["SUPPORTED_DATASETS"] == ["cityscapes", "acdc", "idd"]


def test_supported_backbones(config_file):
    result = load_seg_config(config_file)
    assert result["SUPPORTED_BACKBONES"] == ["pspnet", "segformer"]


def test_primary_metric_default(tmp_path):
    # primary_metric is optional; defaults to mIoU
    cfg = {
        "data_root": "/d",
        "mmconfig_path": "/m",
        "datasets": {"cityscapes": "cityscapes"},
        "supported_backbones": ["pspnet"],
    }
    p = tmp_path / "no_metric.yaml"
    p.write_text(yaml.dump(cfg))
    result = load_seg_config(str(p))
    assert result["PRIMARY_METRIC"] == "mIoU"


def test_schedule_is_optional(config_file):
    # SAMPLE_YAML has no `schedule:` block at all -- the older cluster configs do
    # not, and they must keep loading. train.py falls back to its defaults.
    result = load_seg_config(config_file)
    assert result["SCHEDULE"] == {}


def test_schedule_with_all_values_null_is_empty_not_none(tmp_path):
    """`schedule:` with every value left null parses to None, not {} -- so a plain
    .get default would hand train.py a None to call .get() on."""
    cfg = {
        "data_root": "/d",
        "mmconfig_path": "/m",
        "datasets": {"cityscapes": "cityscapes"},
        "supported_backbones": ["pspnet"],
        "schedule": None,
    }
    p = tmp_path / "null_schedule.yaml"
    p.write_text(yaml.dump(cfg))
    result = load_seg_config(str(p))
    assert result["SCHEDULE"] == {}


def test_schedule_carries_the_two_clocks(tmp_path):
    cfg = {
        "data_root": "/d",
        "mmconfig_path": "/m",
        "datasets": {"cityscapes": "cityscapes"},
        "supported_backbones": ["pspnet"],
        "schedule": {"round_interval": 4000, "corr_interval": 20000},
    }
    p = tmp_path / "schedule.yaml"
    p.write_text(yaml.dump(cfg))
    result = load_seg_config(str(p))
    assert result["SCHEDULE"]["round_interval"] == 4000
    assert result["SCHEDULE"]["corr_interval"] == 20000


def test_missing_required_key_raises(tmp_path):
    # mmconfig_path is required
    cfg = {
        "data_root": "/d",
        "datasets": {"cityscapes": "cityscapes"},
        "supported_backbones": ["pspnet"],
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(KeyError):
        load_seg_config(str(p))


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_seg_config("/nonexistent/path/config.yaml")
