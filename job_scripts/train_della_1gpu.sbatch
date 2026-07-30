#!/bin/bash

#SBATCH --job-name=sensaug-gradcorr-1gpu
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.out
#SBATCH --error=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.err

# config (overridable via env vars):
NAME="${NAME:-sensaug-gradcorr-1gpu}"
# grad_corr = SA over the differentiable ops + the gradient cross-correlation
# pipeline. Add --no-corr-sa below for the control arm (no SA, probes at 0.5).
AUG_TYPE="${1:-grad_corr}"
BACKBONE="${2:-pspnet}"
DATASET="${3:-cityscapes}"
# Name includes AUG_TYPE so different aug types (or none vs grad_corr) never
# collide on the same work_dir -- train.py's resume check would otherwise
# silently continue a previous run's checkpoint under a different config.
# Keep "gradcorr" in the name: train.py appends "_gradcorr" (or "_gradcorr_nosa")
# to any exp_name lacking it when --aug-type=grad_corr, which would otherwise
# desync EXP_NAME from the real work_dir -- and from the resume guard below.
# RUN_TAG suffixes the name for repeat runs of the same config, e.g. RUN_TAG=seed2.
EXP_NAME="${4:-${AUG_TYPE}_${BACKBONE}_${DATASET}_1gpu${RUN_TAG:+_${RUN_TAG}}}"
WORK_DIR="${WORK_DIR:-./experiments}"
CONFIG="${CONFIG:-configs/della.yaml}"

cd /projects/PUCHALLA/LLP2024/tumor

# train.py silently resumes when work_dir/last_checkpoint exists -- no --resume needed.
# Fail loudly instead, so a resubmit never continues a previous run by accident.
# Set ALLOW_RESUME=1 to intentionally continue an existing experiment.
if [ -z "${ALLOW_RESUME}" ] && [ -f "${WORK_DIR}/${EXP_NAME}/last_checkpoint" ]; then
    echo "ERROR: ${WORK_DIR}/${EXP_NAME} already holds a checkpoint; train.py would resume it." >&2
    echo "Pass a fresh exp name (4th arg) or RUN_TAG=..., or set ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

# conda init in ~/.bashrc is stale (points at a removed micromamba); activate the
# env directly by putting its bin on PATH so python + torchrun resolve correctly.
export CONDA_PREFIX=/home/am0532/.conda/envs/sensaug
export PATH=${CONDA_PREFIX}/bin:$PATH

module load cudatoolkit/12.6 2>/dev/null || true

# Use the GPUs SLURM actually allocated to this job, NOT nvidia-smi (which lists every
# physical GPU on the node and would over-count on shared multi-GPU nodes, spawning
# torchrun ranks with no device -> "CUDA error: invalid device ordinal").
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Number of GPUS: $NUM_GPUS"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Let SLURM's CUDA_VISIBLE_DEVICES stand: it already scopes us to the allocated GPU, and
# on MIG nodes it holds a MIG UUID that an integer index would silently clobber.
OMP_NUM_THREADS=4 \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --cluster-config=${CONFIG} \
    --backbone=${BACKBONE} \
    --dataset=${DATASET} \
    --aug-type=${AUG_TYPE} \
    --work_dir=${WORK_DIR} \
    --exp_name=${EXP_NAME} \
    --launcher="pytorch" \
    --no-inv-aug \
    --auto-scale-lr
