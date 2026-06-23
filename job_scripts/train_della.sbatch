#!/bin/bash

#SBATCH --job-name=sensaug
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=../logs/%x/%x.%j.out
#SBATCH --error=../logs/%x/%x.%j.err

# config (overridable via env vars):
NAME="${NAME:-sensaug}"
AUG_TYPE="${1:-none}"
BACKBONE="${2:-pspnet}"
DATASET="${3:-cityscapes}"
EXP_NAME="${4:-${AUG_TYPE}_${BACKBONE}_${DATASET}}"
WORK_DIR="${WORK_DIR:-./experiments}"
CONFIG="${CONFIG:-configs/della.yaml}"

mkdir -p ../logs/${NAME}

cd "$SLURM_SUBMIT_DIR/.."

eval "$(conda shell.bash hook)"
conda activate sensaug

module load cuda/11.7.0

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUS: $NUM_GPUS"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
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
