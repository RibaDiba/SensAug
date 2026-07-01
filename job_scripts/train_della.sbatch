#!/bin/bash

#SBATCH --job-name=sensaug-19
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4

#SBATCH --output=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.out
#SBATCH --error=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.err

# config (overridable via env vars):
NAME="${NAME:-sensaug-19}"
AUG_TYPE="${1:-ours}"
BACKBONE="${2:-pspnet}"
DATASET="${3:-cityscapes}"
EXP_NAME="${4:-${AUG_TYPE}_${BACKBONE}_${DATASET}}"
WORK_DIR="${WORK_DIR:-./experiments}"
CONFIG="${CONFIG:-configs/della.yaml}"

cd /projects/PUCHALLA/LLP2024/tumor

# conda init in ~/.bashrc is stale (points at a removed micromamba); activate the
# env directly by putting its bin on PATH so python + torchrun resolve correctly.
export CONDA_PREFIX=/home/am0532/.conda/envs/sensaug
export PATH=${CONDA_PREFIX}/bin:$PATH

module load cudatoolkit/12.6 2>/dev/null || true

# Use the GPUs SLURM actually allocated to this job, NOT nvidia-smi (which lists every
# physical GPU on the node and would over-count on shared multi-GPU nodes, spawning
# torchrun ranks with no device -> "CUDA error: invalid device ordinal").
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "Number of GPUS: $NUM_GPUS"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=16 \
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
    --no-inv-aug
