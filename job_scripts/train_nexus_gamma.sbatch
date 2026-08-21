#!/bin/bash

#SBATCH --job-name=sensaug_gamma
#SBATCH --nodes=1

## GAMMA training config (per the lab's train_generic.sh guidelines)
#SBATCH --time=48:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:4
#SBATCH --mem=120gb
#SBATCH --ntasks=16

## Scavenger fallback -- swap the four gamma lines above for these when gamma is full.
##SBATCH --time=48:00:00
##SBATCH --qos=scavenger
##SBATCH --account=scavenger
##SBATCH --partition=scavenger

## Tron fallback (shorter walltime, explicit GPU model)
##SBATCH --time=23:59:00
##SBATCH --qos=high
##SBATCH --account=nexus
##SBATCH --partition=tron
##SBATCH --gres=gpu:rtxa4000:4
##SBATCH --mem=64gb
##SBATCH --ntasks=15

# NOTE: SLURM opens these files *before* the script body runs, so the directory
# must already exist at submit time -- the `mkdir -p logs` further down is too
# late for these two. Create it once:  mkdir -p /fs/gammascratch/amodak/sensaug/logs
#SBATCH --output=/fs/gammascratch/amodak/sensaug/logs/%x.%j.out
#SBATCH --error=/fs/gammascratch/amodak/sensaug/logs/%x.%j.err

## failure notifications -- uncomment and fill in
##SBATCH --mail-user=amodak@terpmail.umd.edu
##SBATCH --mail-type=FAIL,END

set -o pipefail

# ---------------------------------------------------------------- config ----
# All overridable via env vars.
NAME="${NAME:-sensaug_gamma}"

# Valid --aug-type values (train.py argparse choices):
#   none ours default grad_corr random autoaugment augmix randaugment
#   trivialaugment idbh vip
# "sensaug" is NOT one of them -- train.py rejects it at argparse before any
# setup runs. The sensitivity-informed arm is `ours`; `grad_corr` additionally
# runs the gradient cross-correlation pipeline (add --no-corr-sa for its control
# arm: no SA, probes at the fixed 0.5 reference magnitude).
AUG_TYPE="${1:-none}"
BACKBONE="${2:-segformer}"
DATASET="${3:-cityscapes}"

# Validate here rather than letting train.py's argparse catch it: under
# --launcher=pytorch that error prints once per rank (4x on this script) and
# only after torchrun has already spun up, which is a noisy and roundabout
# way to learn AUG_TYPE was misspelled.
VALID_AUG_TYPES=(none ours grad_corr default random autoaugment augmix randaugment trivialaugment idbh vip)
aug_type_ok=0
for valid in "${VALID_AUG_TYPES[@]}"; do
    if [ "${AUG_TYPE}" = "${valid}" ]; then
        aug_type_ok=1
        break
    fi
done
if [ "${aug_type_ok}" -ne 1 ]; then
    echo "ERROR: AUG_TYPE='${AUG_TYPE}' is not a valid --aug-type." >&2
    echo "Valid values: ${VALID_AUG_TYPES[*]}" >&2
    exit 1
fi

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

# Your own checkout. The SBATCH log paths above point at /fs/gammascratch/amodak
# -- if that is also where the repo lives, set REPO_DIR to match rather than
# using the shared lab checkout below.
REPO_DIR="${REPO_DIR:-/fs/nexus-scratch/lyzheng/sensaug}"
cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
if [ ! -f train.py ]; then
    echo "ERROR: no train.py under REPO_DIR=${REPO_DIR}; wrong checkout?" >&2
    exit 1
fi
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
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
echo "Number of GPUS: $NUM_GPUS"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# Derive from the job id so two jobs landing on the same node don't both grab
# 29500 and die with "address already in use".
PORT=${PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TORCHRUN_ARGS=(
    --nnodes=$NNODES
    --node_rank=$NODE_RANK
    --master_addr=$MASTER_ADDR
    --nproc_per_node=$NUM_GPUS
    --master_port=$PORT
)

export OMP_NUM_THREADS=16

# ---------------------------------------------------------------- train ----
# The correlation pipeline is an --aug-type now, not a flag: to enable it here,
# pass grad_corr as the first positional arg (or set AUG_TYPE), rather than
# uncommenting a flag.
torchrun "${TORCHRUN_ARGS[@]}" \
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

TRAIN_STATUS=$?
if [ $TRAIN_STATUS -ne 0 ]; then
    echo "ERROR: training exited ${TRAIN_STATUS}; skipping evaluation." >&2
    exit $TRAIN_STATUS
fi

if [ -n "${SKIP_EVAL}" ]; then
    echo "SKIP_EVAL set; training done, not evaluating."
    exit 0
fi

EXP_DIR="${WORK_DIR}/${EXP_NAME}"

# ------------------------------------------------- robustness evaluation ----
# test_robust.py rebuilds the config from ${EXP_DIR}/*.py, so it takes NO
# --cluster-config (its parser has no such flag and would reject it).
torchrun "${TORCHRUN_ARGS[@]}" \
    test_robust.py \
    --work-dir="${EXP_DIR}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best

# --------------------------------------------- real-world domain shifts ----
# Cityscapes-trained models only -- ACDC / IDD / Dark Zurich / Nighttime Driving
# all assume the 19-class Cityscapes label space.
# Unlike test_robust.py, test.py DOES require --cluster-config (required=True).
if [[ "$DATASET" == "cityscapes" ]]; then
    for TEST_DATA in acdc idd zurich night; do
        echo "=== test.py on ${TEST_DATA} ==="
        torchrun "${TORCHRUN_ARGS[@]}" \
            test.py \
            --cluster-config=${CONFIG} \
            --work-dir="${EXP_DIR}" \
            --launcher="pytorch" \
            --save-vis \
            --use-best \
            --test_data="${TEST_DATA}" \
        || echo "WARNING: test.py --test_data=${TEST_DATA} failed; continuing." >&2
    done
fi

echo "All done: ${EXP_DIR}"
