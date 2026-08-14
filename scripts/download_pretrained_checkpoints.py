#!/usr/bin/env python
"""Stage pretrained ImageNet backbone checkpoints into the local cache dir.

Della compute nodes have no internet access. Several backbone configs under
sensaug/custom_configs/mmseg/ point their backbone.init_cfg at a
download.openmmlab.com URL (segformer, pspnet-rsb, convnext, swin are the ones
known to do this) -- fine on a machine with internet, but train.py crashes deep in
model.init_weights() with a DNS error when that download happens on a compute node.

train.py's build_config() now redirects those URLs to
`<pretrained_cache_dir>/<basename(url)>` (see configs/della.yaml) and fails fast
with a clear message if the file isn't there yet. This script is how you populate
that cache: it scans the config files for a given backbone, extracts every
`checkpoint = "https://...pth"` URL, and downloads whichever aren't already cached.

Run this from somewhere with real internet egress -- a Della LOGIN node (not a
compute node, not an sbatch job), or your own machine followed by
`rsync`/`scp` of the cache dir to Della.

Known limitation: this only catches the `checkpoint = "http..."` + `type="Pretrained"`
pattern. deeplabv3plus uses mmcv's `pretrained='open-mmlab://resnet50_v1c'`
model-zoo shorthand instead of a literal URL, and mae/vit use a different init
mechanism -- none of those are covered here. No failure has been observed for those
backbones yet; revisit if one shows up.

Usage:
    python scripts/download_pretrained_checkpoints.py --backbone segformer
    python scripts/download_pretrained_checkpoints.py --backbone pspnet convnext swin
    python scripts/download_pretrained_checkpoints.py --all
    python scripts/download_pretrained_checkpoints.py --backbone segformer --dry-run
"""

import argparse
import glob
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.cluster_config import load_seg_config

CHECKPOINT_URL_RE = re.compile(r"""checkpoint\s*=\s*["'](https?://[^"']+\.pth)["']""")


def find_urls(mmconfig_path, backbone):
    """
    Find checkpoint URLs declared in a backbone configuration directory.
    
    Parameters:
    	mmconfig_path (str): Root directory containing backbone configurations.
    	backbone (str): Name of the backbone configuration directory.
    
    Returns:
    	list[str]: Sorted unique checkpoint URLs found in Python configuration files.
    
    Raises:
    	FileNotFoundError: If the backbone configuration directory does not exist.
    """
    backbone_dir = os.path.join(mmconfig_path, backbone)
    if not os.path.isdir(backbone_dir):
        raise FileNotFoundError(f"No such backbone config dir: {backbone_dir}")
    urls = set()
    for path in glob.glob(os.path.join(backbone_dir, "*.py")):
        with open(path) as f:
            urls.update(CHECKPOINT_URL_RE.findall(f.read()))
    return sorted(urls)


def human_size(num_bytes):
    """
    Convert a byte count to a rounded human-readable size using binary units.
    
    Parameters:
    	num_bytes (float): The number of bytes to convert.
    
    Returns:
    	str: The rounded size with a B, KB, MB, GB, or TB suffix.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.0f}TB"


def remote_size(url):
    """
    Retrieve the remote file size from its HTTP headers.
    
    Parameters:
    	url (str): URL of the remote file.
    
    Returns:
    	int or None: The file size in bytes, or `None` if the response has no `Content-Length` header.
    """
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def download(url, dest_dir, dry_run):
    """
    Download a checkpoint into the cache directory unless it is already cached or dry-run mode is enabled.
    
    Parameters:
    	url (str): HTTPS checkpoint URL.
    	dest_dir (str): Directory where the checkpoint is cached.
    	dry_run (bool): Whether to report the download without retrieving the file.
    """
    dest = os.path.join(dest_dir, os.path.basename(url))
    if os.path.isfile(dest):
        print(f"  [cached]     {os.path.basename(dest)}")
        return
    if dry_run:
        try:
            size = remote_size(url)
            size_str = human_size(size) if size is not None else "unknown size"
        except OSError as e:
            size_str = f"HEAD failed ({e})"
        print(f"  [would fetch] {url} ({size_str})")
        return
    print(f"  [fetching]   {url}")
    os.makedirs(dest_dir, exist_ok=True)
    tmp_dest = dest + ".partial"
    urllib.request.urlretrieve(url, tmp_dest)
    os.rename(tmp_dest, dest)
    print(f"  [done]       {dest}")


def main():
    """Run the checkpoint staging command for selected or all supported backbones.
    
    Reads the cluster configuration, scans each selected backbone for checkpoint URLs, and downloads missing checkpoints to the configured cache directory. In dry-run mode, reports the checkpoints that would be downloaded without fetching them.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backbone",
        nargs="+",
        default=None,
        help="one or more backbone dir names under mmconfig_path, e.g. segformer pspnet",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan every supported_backbones entry from --cluster-config instead of --backbone",
    )
    parser.add_argument(
        "--cluster-config",
        default="configs/della.yaml",
        help="YAML cluster config to source mmconfig_path / pretrained_cache_dir from (default: configs/della.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="override the cache dir instead of reading pretrained_cache_dir from --cluster-config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be downloaded (with sizes) without fetching anything",
    )
    args = parser.parse_args()

    if not args.backbone and not args.all:
        parser.error("pass --backbone <name>... or --all")

    seg = load_seg_config(args.cluster_config)
    mmconfig_path = seg["MMCONFIG_PATH"]
    cache_dir = args.cache_dir or seg["PRETRAINED_CACHE_DIR"]
    if not cache_dir:
        parser.error(
            "no cache dir: pass --cache-dir, or set pretrained_cache_dir in "
            f"{args.cluster_config}"
        )

    backbones = seg["SUPPORTED_BACKBONES"] if args.all else args.backbone

    for backbone in backbones:
        print(f"{backbone}:")
        urls = find_urls(mmconfig_path, backbone)
        if not urls:
            print("  (no checkpoint URLs found in this backbone's configs)")
            continue
        for url in urls:
            download(url, cache_dir, args.dry_run)


if __name__ == "__main__":
    main()
