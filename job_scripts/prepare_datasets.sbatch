#!/bin/bash
# Download + convert the PUBLIC datasets (pascal_voc12, loveda) on a compute/data
# node. nexus.yaml has no pretrained_cache_dir, which per its own comment means
# node-level internet egress is assumed -- so the downloads work from a batch job.
#
# The GATED datasets (potsdam, synapse, a2i2haze, acdc, idd) are NOT done here:
# they need a manual, logged-in download first. Run
#   python scripts/prepare_datasets.py <key> --src <archive> --cluster-config configs/nexus.yaml
# on a login node once you have the archive.
#
# Override the set with:  DATASETS="pascal_voc12" sbatch job_scripts/prepare_datasets.sbatch

#SBATCH --job-name=prepare-datasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --qos=default
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.err
#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -euo pipefail

REPO=/gammascratch/amodak/SensAug
cd "$REPO"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-sensaug}"

DATASETS="${DATASETS:-pascal_voc12 loveda}"

echo "Preparing: $DATASETS"
python scripts/prepare_datasets.py $DATASETS --cluster-config configs/nexus.yaml

echo "---- final state ----"
python scripts/prepare_datasets.py --all --cluster-config configs/nexus.yaml --check
