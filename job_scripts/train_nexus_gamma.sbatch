#!/bin/bash

#SBATCH --job-name=sensaug_gamma
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma

#SBATCH --output=/fs/nexus-scratch/lyzheng/sensaug/logs/%x.%j.out
#SBATCH --error=/fs/nexus-scratch/lyzheng/sensaug/logs/%x.%j.err

# config (overridable via env vars):
NAME="${NAME:-sensaug_gamma}"
# grad_corr = SA over the differentiable ops + the gradient cross-correlation
# pipeline. Add --no-corr-sa below for the control arm (no SA, probes at 0.5).
AUG_TYPE="${1:-default}"
BACKBONE="${2:-segformer}"
DATASET="${3:-cityscapes}"
# Name includes AUG_TYPE so different aug types (or none vs grad_corr) never
# collide on the same work_dir -- train.py's resume check would otherwise
# silently continue a previous run's checkpoint under a different config.
# Keep "ours" in the name: train.py appends "_ours" to any exp_name lacking it when
# --aug-type=ours, which would desync EXP_NAME from the real work_dir (and the guard below).
# RUN_TAG suffixes the name for repeat runs of the same config, e.g. RUN_TAG=seed2.
EXP_NAME="${4:-${AUG_TYPE}_${NAME}_${BACKBONE}_${DATASET}_4gpu${RUN_TAG:+_${RUN_TAG}}}"
WORK_DIR="${WORK_DIR:-./experiments}"
# gamma is a UMD Nexus partition/account -- use the Nexus cluster config, not della's.
CONFIG="${CONFIG:-configs/nexus.yaml}"

# Which function turns the redundancy score into the reweighted training pdf.
# train.py REQUIRES it on every --aug-type=grad_corr run (there is no default, so
# an arm is never left unnamed) and warns that it is ignored on every other arm --
# hence conditional rather than always appended.
DOWNWEIGHT="${DOWNWEIGHT:-soft-weighting}"
EXTRA_ARGS=()
if [ "${AUG_TYPE}" = "grad_corr" ]; then
    EXTRA_ARGS+=(--corr-downweight-method="${DOWNWEIGHT}")
fi

# Shared group checkout on Nexus scratch (matches configs/nexus.yaml's mmconfig_path).
# Override REPO_DIR if your own checkout lives elsewhere.
REPO_DIR="${REPO_DIR:-/fs/nexus-scratch/lyzheng/sensaug}"
cd "${REPO_DIR}"
mkdir -p logs

# train.py silently resumes when work_dir/last_checkpoint exists -- no --resume needed.
# Fail loudly instead, so a resubmit never continues a previous run by accident.
# Set ALLOW_RESUME=1 to intentionally continue an existing experiment.
if [ -z "${ALLOW_RESUME}" ] && [ -f "${WORK_DIR}/${EXP_NAME}/last_checkpoint" ]; then
    echo "ERROR: ${WORK_DIR}/${EXP_NAME} already holds a checkpoint; train.py would resume it." >&2
    echo "Pass a fresh exp name (4th arg) or RUN_TAG=..., or set ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-sensaug}"

module load cuda/11.7.0

# Nexus compute nodes have internet access (unlike della's), so no pretrained-checkpoint
# pre-staging is needed here -- see sensaug/cluster_config.py.
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "Number of GPUS: $NUM_GPUS"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

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
    --auto-scale-lr \
    --descending-MA \
    "${EXTRA_ARGS[@]}"
# The correlation pipeline is an --aug-type now, not a flag: to enable it here,
# pass grad_corr as the first positional arg (or set AUG_TYPE), rather than
# uncommenting a flag.
