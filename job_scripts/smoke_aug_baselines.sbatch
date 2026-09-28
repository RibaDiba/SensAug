#!/bin/bash

# Smoke test for one (--aug-type, dataset) pair: a handful of training iterations
# on a single GPU of any type, then one full val pass. Proves the pipeline builds,
# the augmentation transform runs on real data with its label, and train + val
# both complete -- not that anything trains well.
#
#   sbatch job_scripts/smoke_aug_baselines.sbatch <aug-type> <dataset> [backbone]
#
# Env: MAX_ITERS (default 20), CONDA_ENV, REPO_DIR, WORK_DIR.

#SBATCH --job-name=smoke-aug
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/smoke/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/smoke/%x.%j.err

set -o pipefail

AUG_TYPE="${1:?usage: smoke_aug_baselines.sbatch <aug-type> <dataset> [backbone]}"
DATASET="${2:?usage: smoke_aug_baselines.sbatch <aug-type> <dataset> [backbone]}"
BACKBONE="${3:-segformer}"
MAX_ITERS="${MAX_ITERS:-20}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/experiments/smoke}"
EXP_NAME="smoke_${AUG_TYPE}_${BACKBONE}_${DATASET}_${SLURM_JOB_ID:-local}"

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }

# `module` is not inherited by a script-submitted job; conda.sh works without it.
command -v conda >/dev/null 2>&1 || \
    source /opt/common/miniforge3/latest/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-sensaug}" || { echo "ERROR: conda activate failed" >&2; exit 1; }
command -v torchrun >/dev/null 2>&1 || { echo "ERROR: torchrun not on PATH" >&2; exit 1; }

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "aug=${AUG_TYPE} dataset=${DATASET} backbone=${BACKBONE} max_iters=${MAX_ITERS}"

export OMP_NUM_THREADS=4
PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

# Same flags as train_nexus_gamma.sbatch, on one process. --round_interval set
# to MAX_ITERS so the single val pass lands on the last iteration.
torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=${PORT} \
    train.py \
    --cluster-config=configs/nexus.yaml \
    --backbone=${BACKBONE} \
    --dataset=${DATASET} \
    --aug-type=${AUG_TYPE} \
    --work_dir=${WORK_DIR} \
    --exp_name=${EXP_NAME} \
    --launcher="pytorch" \
    --no-inv-aug \
    --auto-scale-lr \
    --descending-MA \
    --max-iters=${MAX_ITERS} \
    --round_interval=${MAX_ITERS}
STATUS=$?

echo "SMOKE_RESULT aug=${AUG_TYPE} dataset=${DATASET} exit=${STATUS}"
exit $STATUS
