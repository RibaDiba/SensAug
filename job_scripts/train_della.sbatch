#!/bin/bash

#SBATCH --job-name=sensaug-print-matrix-2-4-gpus
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
# della-l01g[1-12] are MIG-sliced (3g.40gb / 1g.10gb) and match a bare --gres=gpu:4, but
# several slices share one physical A100. NCCL keys devices by PCI bus id and has no P2P
# between MIG instances -> "Duplicate GPU detected". nomig keeps us on whole-GPU nodes.
#SBATCH --constraint=nomig

#SBATCH --output=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.out
#SBATCH --error=/projects/PUCHALLA/LLP2024/tumor/logs/%x.%j.err

# config (overridable via env vars):
NAME="${NAME:-sensaug-print-matrix-2-4-gpus}"
# grad_corr = SA over the differentiable ops + the gradient cross-correlation
# pipeline. Add --no-corr-sa below for the control arm (no SA, probes at 0.5).
AUG_TYPE="${1:-grad_corr}"
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

# Let SLURM's CUDA_VISIBLE_DEVICES stand: it already scopes us to the allocated GPUs, and
# on MIG nodes it holds MIG UUIDs that an integer list would silently clobber.
OMP_NUM_THREADS=16 \
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
