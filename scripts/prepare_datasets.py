#!/usr/bin/env python
"""Install the segmentation datasets registered in a cluster config.

`configs/nexus.yaml` lists 9 datasets under `datasets:`. cluster_config.py turns
each into `DATA_ROOT_LOOKUP[key] = "<data_root>/<value>"`, and train.py / test.py
then trust that a correctly-laid-out tree exists there. This script puts that tree
on disk.

Two classes of dataset:

  * PUBLIC  -- pascal_voc12, loveda: direct download URLs, installed end to end
              with no human step.
  * GATED   -- potsdam, synapse, a2i2haze, acdc, idd: the source is behind a
              registration wall (or, for a2i2haze, is only shared privately). Pass
              `--src /path/to/download` once you have the archive; with no --src
              the script prints exactly where to get it and stops -- it does not
              raise.

Idempotent: a dataset whose expected tree already exists is skipped. Safe to
re-run after fetching more archives.

Usage:
    # everything that needs no login:
    python scripts/prepare_datasets.py pascal_voc12 loveda --cluster-config configs/nexus.yaml

    # a gated one, after downloading its archive(s):
    python scripts/prepare_datasets.py acdc --cluster-config configs/nexus.yaml \
        --src /gammascratch/amodak/downloads/acdc

    # just print where to get the gated archives:
    python scripts/prepare_datasets.py potsdam synapse acdc idd a2i2haze \
        --cluster-config configs/nexus.yaml

    # check what is already installed without downloading anything:
    python scripts/prepare_datasets.py --all --cluster-config configs/nexus.yaml --check
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

import yaml
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensaug.cluster_config import load_seg_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTERS = os.path.join(REPO_ROOT, "sensaug", "custom_configs", "dataset_converters")

# MMSeg's canonical Synapse (BTCV) volume split -- 18 train / 12 val, 30 total.
# synapse.py's converter does `line[3:7]` on each entry, so lines are "imgNNNN.nii.gz".
SYNAPSE_TRAIN = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40]
SYNAPSE_VAL = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]


# --------------------------------------------------------------------------- io

def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}PB"


def download(url, dest):
    """Fetch `url` to `dest`, resuming a previous `.partial` where the server allows."""
    if os.path.isfile(dest):
        print(f"    [cached]  {os.path.basename(dest)} ({_human(os.path.getsize(dest))})")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".partial"
    have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    print(f"    [fetch]   {url}")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if have and e.code == 416:  # already complete
            resp = None
        else:
            raise
    if resp is not None:
        mode = "ab" if resp.status == 206 and have else "wb"
        if mode == "wb":
            have = 0
        tty = sys.stderr.isatty()
        with open(tmp, mode) as f:
            total = have + int(resp.headers.get("Content-Length", 0) or 0)
            done = have
            next_mark = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if not total:
                    continue
                pct = 100 * done / total
                if tty:
                    print(f"\r      {_human(done)}/{_human(total)} ({pct:4.1f}%)",
                          end="", flush=True)
                elif pct >= next_mark:  # log file: one line per 5%
                    print(f"      {_human(done)}/{_human(total)} ({pct:4.1f}%)", flush=True)
                    next_mark += 5
        print()
    os.rename(tmp, dest)
    print(f"    [done]    {dest} ({_human(os.path.getsize(dest))})")
    return dest


def download_first(urls, dest):
    """Try each URL in turn; keep the first that works."""
    last = None
    for url in urls:
        try:
            return download(url, dest)
        except Exception as e:  # noqa: BLE001 - fall through to the next mirror
            last = e
            print(f"    [warn]    {url} failed: {e}")
    raise RuntimeError(f"all mirrors failed for {os.path.basename(dest)}: {last}")


def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"    [extract] {os.path.basename(archive)} -> {dest}")
    if archive.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        raise ValueError(f"don't know how to extract {archive}")


def run_converter(script, *cli_args):
    path = os.path.join(CONVERTERS, script)
    cmd = [sys.executable, path, *map(str, cli_args)]
    print(f"    [convert] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def count(pattern):
    return len(glob.glob(pattern, recursive=True))


def find_dir(root, name):
    """First directory called `name` anywhere under `root` (archives nest unpredictably)."""
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None


# --------------------------------------------------------------- dataset specs

class Dataset:
    key = ""
    public = False
    # relpath under the dataset's data_root -> (approx expected file count, glob)
    expected = {}
    instructions = ""

    def verify(self, root):
        rows = []
        ok = True
        for rel, (approx, pat) in self.expected.items():
            n = count(os.path.join(root, pat))
            good = n > 0 and (approx is None or n >= 0.5 * approx)
            ok = ok and good
            want = "any" if approx is None else f"~{approx}"
            rows.append(f"      {'ok ' if good else 'MISS'} {rel:<24} found {n:>7}  (want {want})")
        return ok, rows

    def already_installed(self, root):
        ok, _ = self.verify(root)
        return ok

    def install(self, root, src, downloads):
        raise NotImplementedError


class PascalVOC12(Dataset):
    key = "pascal_voc12"
    public = True
    # target root is <data_root>/VOCdevkit/VOC2012 ; the tar unpacks VOCdevkit/ one
    # level up, so we extract into the parent of the parent.
    expected = {
        "JPEGImages": (17125, "JPEGImages/*.jpg"),
        "SegmentationClass": (2913, "SegmentationClass/*.png"),
        "ImageSets/Segmentation/train.txt": (None, "ImageSets/Segmentation/train.txt"),
        "ImageSets/Segmentation/val.txt": (None, "ImageSets/Segmentation/val.txt"),
    }
    urls = [
        "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "https://data.pjreddie.com/files/VOCtrainval_11-May-2012.tar",
    ]

    def install(self, root, src, downloads):
        # root == <data_root>/VOCdevkit/VOC2012 ; unpack VOCdevkit/ into <data_root>
        data_root = os.path.dirname(os.path.dirname(root))
        tar = src or download_first(self.urls, os.path.join(downloads, "VOCtrainval_11-May-2012.tar"))
        extract(tar, data_root)


class LoveDA(Dataset):
    key = "loveda"
    public = True
    expected = {
        "img_dir/train": (2522, "img_dir/train/*.png"),
        "ann_dir/train": (2522, "ann_dir/train/*.png"),
        "img_dir/val": (1669, "img_dir/val/*.png"),
        "ann_dir/val": (1669, "ann_dir/val/*.png"),
    }
    zenodo = "https://zenodo.org/records/5706578/files"

    def install(self, root, src, downloads):
        stage = src or downloads
        stage = os.path.join(stage, "loveda")
        os.makedirs(stage, exist_ok=True)
        for name in ("Train.zip", "Val.zip", "Test.zip"):
            if not os.path.isfile(os.path.join(stage, name)):
                download(f"{self.zenodo}/{name}?download=1", os.path.join(stage, name))
        run_converter("loveda.py", stage, "-o", root)


class Potsdam(Dataset):
    key = "potsdam"
    public = False
    expected = {
        "img_dir/train": (3456, "img_dir/train/*.png"),
        "ann_dir/train": (3456, "ann_dir/train/*.png"),
        "img_dir/val": (2016, "img_dir/val/*.png"),
        "ann_dir/val": (2016, "ann_dir/val/*.png"),
    }
    instructions = (
        "ISPRS 2D Semantic Labeling Contest -- Potsdam.\n"
        "  1. Request access:  https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx\n"
        "     (or the Uni Hannover mirror linked from that page).\n"
        "  2. You need two archives:  2_Ortho_RGB.zip  and  5_Labels_all.zip\n"
        "  3. Put both in one directory and re-run with:\n"
        "       --src /path/to/that/directory\n"
    )

    def install(self, root, src, downloads):
        zips = glob.glob(os.path.join(src, "*.zip"))
        assert any("Ortho_RGB" in z for z in zips), f"2_Ortho_RGB.zip not found in {src}"
        assert any("Labels_all" in z for z in zips), f"5_Labels_all.zip not found in {src}"
        run_converter("potsdam.py", src, "-o", root,
                      "--clip_size", 512, "--stride_size", 512)


class Synapse(Dataset):
    key = "synapse"
    public = False
    expected = {
        "img_dir/train": (2212, "img_dir/train/*.jpg"),
        "ann_dir/train": (2212, "ann_dir/train/*.png"),
        "img_dir/val": (1568, "img_dir/val/*.jpg"),
        "ann_dir/val": (1568, "ann_dir/val/*.png"),
    }
    instructions = (
        "Synapse multi-organ (BTCV 'Multi-Atlas Labeling Beyond the Cranial Vault').\n"
        "  1. Make a synapse.org account, then open project  syn3193805\n"
        "     https://www.synapse.org/#!Synapse:syn3193805/wiki/217789\n"
        "  2. Download 'Abdomen/RawData.zip' (the 30 Training img/label .nii.gz volumes).\n"
        "  3. Extract it and re-run with --src pointing at the folder that\n"
        "     contains  img/imgNNNN.nii.gz  and  label/labelNNNN.nii.gz\n"
        "     (i.e. RawData/Training). The train/val split is written for you.\n"
        "  Needs:  pip install nibabel\n"
    )

    def install(self, root, src, downloads):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            sys.exit("synapse conversion needs nibabel -- `pip install nibabel` and re-run")
        img_dir = find_dir(src, "img")
        label_dir = find_dir(src, "label")
        if not (img_dir and label_dir):
            sys.exit(f"could not find img/ and label/ under {src}")
        stage = os.path.join(downloads, "synapse_stage")
        os.makedirs(stage, exist_ok=True)
        for sub, real in (("img", img_dir), ("label", label_dir)):
            dst = os.path.join(stage, sub)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(real), dst)
        with open(os.path.join(stage, "train.txt"), "w") as f:
            f.writelines(f"img{n:04d}.nii.gz\n" for n in SYNAPSE_TRAIN)
        with open(os.path.join(stage, "val.txt"), "w") as f:
            f.writelines(f"img{n:04d}.nii.gz\n" for n in SYNAPSE_VAL)
        run_converter("synapse.py", "--dataset-path", stage, "--save-path", root)


class A2I2Haze(Dataset):
    key = "a2i2haze"
    public = False
    expected = {
        "imgs/train": (None, "imgs/train/*c.jpg"),
        "labels/train": (None, "labels/train/*c_labelTrainIds.png"),
        "imgs/val": (None, "imgs/val/*c.jpg"),
        "labels/val": (None, "labels/val/*c_labelTrainIds.png"),
    }
    instructions = (
        "A2I2-Haze has no public MMSeg recipe. Ask the codebase contact\n"
        "(Laura Zheng <lyzheng@umd.edu>) for the post-processed zip.\n"
        "  Expected inside --src (a dir or a .zip):\n"
        "    imgs/{train,val}/*c.jpg\n"
        "    labels/{train,val}/*c_labelTrainIds.png\n"
        "  Re-run with --src /path/to/zip-or-dir\n"
    )

    def install(self, root, src, downloads):
        if os.path.isfile(src) and src.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            tmp = os.path.join(downloads, "a2i2haze_x")
            extract(src, tmp)
            src = find_dir(tmp, "imgs") and os.path.dirname(find_dir(tmp, "imgs")) or tmp
        os.makedirs(root, exist_ok=True)
        for sub in ("imgs", "labels"):
            s = os.path.join(src, sub)
            if not os.path.isdir(s):
                sys.exit(f"{s} missing -- see the layout above")
            shutil.copytree(s, os.path.join(root, sub), dirs_exist_ok=True)


class ACDC(Dataset):
    key = "acdc"
    public = False
    # test.py reads rgb_anno/test + gt/test ; the converter (--split all) also fills train.
    expected = {
        "rgb_anno/train": (1600, "rgb_anno/train/*_rgb_anon.png"),
        "gt/train": (1600, "gt/train/*_gt_labelTrainIds.png"),
        "rgb_anno/test": (2000, "rgb_anno/test/*_rgb_anon.png"),
        "gt/test": (406, "gt/test/*_gt_labelTrainIds.png"),
    }
    instructions = (
        "ACDC (Adverse Conditions Dataset with Correspondences).\n"
        "  1. Register:  https://acdc.vision.ee.ethz.ch/\n"
        "  2. Download  rgb_anon_trainvaltest.zip  and  gt_trainval.zip\n"
        "  3. Put both (or their extracted trees) in one directory and re-run with:\n"
        "       --src /path/to/that/directory\n"
        "  The vendored converter (sensaug/custom_configs/dataset_converters/acdc.py,\n"
        "  --split all) maps the raw val/ split to test/.\n"
    )

    def install(self, root, src, downloads):
        stage = src
        zips = glob.glob(os.path.join(src, "*.zip"))
        if zips:
            stage = os.path.join(downloads, "acdc_raw")
            for z in zips:
                extract(z, stage)
        run_converter("acdc.py", stage, "-o", root, "--split", "all")


class IDD(Dataset):
    key = "idd"
    public = False
    # test.py:apply_idd_eval -> leftImg8bit/val + gtFine/val, cityscapes-style TrainIds
    expected = {
        "leftImg8bit/val": (2036, "leftImg8bit/val/**/*_leftImg8bit.png"),
        "gtFine/val": (2036, "gtFine/val/**/*_gtFine_labelTrainIds.png"),
    }
    instructions = (
        "IDD (India Driving Dataset) -- Segmentation (IDD 20k Part I).\n"
        "  1. Register:  https://idd.insaan.iiit.ac.in/  ->  'IDD Segmentation (IDD 20k Part I)'\n"
        "  2. Extract  idd-segmentation.tar.gz  -> gives  IDD_Segmentation/{leftImg8bit,gtFine}/{train,val}/<seq>/\n"
        "  3. Generate cityscapes-style train-id labels with AutoNUE's public code:\n"
        "       git clone https://github.com/AutoNUE/public-code\n"
        "       export PYTHONPATH=public-code/helpers\n"
        "       python public-code/preperation/createLabels.py \\\n"
        "           --datadir IDD_Segmentation --id-type level3Id --num-workers 8\n"
        "     then rename the produced *_gtFine_labellevel3Ids.png -> *_gtFine_labelTrainIds.png\n"
        "     (level3Id is the 19-class Cityscapes-compatible id set).\n"
        "  4. Re-run with --src pointing at that IDD_Segmentation dir.\n"
    )

    def install(self, root, src, downloads):
        base = src
        if os.path.isfile(src) and src.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
            base = os.path.join(downloads, "idd_raw")
            extract(src, base)
        li = find_dir(base, "leftImg8bit")
        gt = find_dir(base, "gtFine")
        if not (li and gt):
            sys.exit(f"leftImg8bit/ and gtFine/ not found under {src}")
        if not count(os.path.join(gt, "val", "**", "*_gtFine_labelTrainIds.png")):
            sys.exit(
                "no *_gtFine_labelTrainIds.png under gtFine/val -- run AutoNUE's "
                "createLabels.py first (see the instructions above)"
            )
        os.makedirs(root, exist_ok=True)
        for sub, real in (("leftImg8bit", li), ("gtFine", gt)):
            dst = os.path.join(root, sub)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(real), dst)


DATASETS = {
    d.key: d
    for d in (PascalVOC12(), LoveDA(), Potsdam(), Synapse(), A2I2Haze(), ACDC(), IDD())
}


# ---------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="*", metavar="DATASET",
                    help="dataset keys to install: " + ", ".join(DATASETS))
    ap.add_argument("--all", action="store_true", help="every gated+public dataset")
    ap.add_argument("--cluster-config", default="configs/nexus.yaml")
    ap.add_argument("--src", default=None,
                    help="path to a pre-downloaded archive/dir for a GATED dataset "
                         "(only meaningful with a single dataset arg)")
    ap.add_argument("--downloads-dir", default=None,
                    help="where public archives are cached (default <data_root>/_downloads)")
    ap.add_argument("--check", action="store_true",
                    help="only report installed / missing, download nothing")
    args = ap.parse_args()

    seg = load_seg_config(args.cluster_config)
    lookup = seg["DATA_ROOT_LOOKUP"]
    with open(args.cluster_config) as f:
        data_root = yaml.safe_load(f)["data_root"]
    downloads = args.downloads_dir or os.path.join(data_root, "_downloads")

    unknown = [k for k in args.datasets if k not in DATASETS]
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(unknown)} -- choose from {', '.join(DATASETS)}")
    keys = list(DATASETS) if args.all else args.datasets
    if not keys:
        ap.error("name one or more datasets, or pass --all")
    if args.src and len(keys) != 1:
        ap.error("--src only makes sense with exactly one dataset")

    results = []
    for key in keys:
        ds = DATASETS[key]
        root = lookup.get(key)
        if root is None:
            print(f"\n== {key}: not in {args.cluster_config} datasets: -- skipping")
            continue
        print(f"\n== {key}  ->  {root}")

        if ds.already_installed(root):
            print("   already installed:")
            _, rows = ds.verify(root)
            print("\n".join(rows))
            results.append((key, "already"))
            continue

        if args.check:
            ok, rows = ds.verify(root)
            print("\n".join(rows))
            results.append((key, "ok" if ok else "MISSING"))
            continue

        if not ds.public and not args.src:
            print("   GATED -- needs a manual download:\n")
            print("   " + ds.instructions.replace("\n", "\n   ").rstrip())
            results.append((key, "needs --src"))
            continue

        try:
            ds.install(root, args.src, downloads)
        except subprocess.CalledProcessError as e:
            print(f"   converter failed: {e}")
            results.append((key, "FAILED"))
            continue
        except SystemExit as e:
            print(f"   {e}")
            results.append((key, "FAILED"))
            continue

        ok, rows = ds.verify(root)
        print("\n".join(rows))
        results.append((key, "installed" if ok else "installed? (counts off)"))

    print("\n" + "=" * 60)
    for key, status in results:
        print(f"  {key:<16} {status}")
    print("=" * 60)
    gated_pending = [k for k, s in results if s == "needs --src"]
    if gated_pending:
        print("\nStill need a login / manual download:", ", ".join(gated_pending))
        print("Re-run each with --src once you have its archive.")


if __name__ == "__main__":
    main()
