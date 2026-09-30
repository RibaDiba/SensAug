#!/bin/bash

#SBATCH --job-name=mrmrprune
#SBATCH --nodes=1

## GAMMA training config (per the lab's train_generic.sh guidelines)
#SBATCH --time=36:00:00
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

# NOTE: SLURM opens these files *before* the script body runs, so a %x-named
# subdirectory can't be relied on here -- it wouldn't exist yet for a job name
# picked at submit time via `--job-name=`. These paths are absolute (not
# relative to $SLURM_SUBMIT_DIR) specifically so `sbatch` works the same
# whether you run it from the repo root or from inside job_scripts/ --
# a relative path here previously doubled into job_scripts/job_scripts/logs/
# when submitted from job_scripts/. Bootstrap flat into job_scripts/logs/;
# once running, the script `mv`s these two files into a per-job-name
# subdirectory it creates on the fly -- a same-filesystem rename keeps the
# already-open fd writing to the same inode, so no output is lost and no
# `exec` redirect is needed. Every submission ends up grouped under
# job_scripts/logs/<job-name>/ regardless of job name or submit dir.
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# train_mrmr_prune_retrain.sbatch -- two-stage "mRMR-prune then retrain".
#
# A stage-aware clone of train_nexus_gamma.sbatch:
#
#   STAGE 1 (default)  --aug-type=grad_corr  --corr-downweight-method=mRMR
#                      Trains the mRMR hard-pruning arm exactly like
#                      `train_nexus_gamma.sbatch grad_corr <bb> <ds>` does, then
#                      runs the same robustness eval. Afterwards it scrapes the
#                      final set of ops mRMR drove to probability 0 out of the
#                      run's mmengine log, writes them to
#                      <work_dir>/<exp>/mrmr_pruned_ops.txt, and submits STAGE 2
#                      as a SLURM dependent job (--dependency=afterok).
#
#   STAGE 2            --aug-type=ours  --pruned-augmentations <ops from stage 1>
#                      A plain sensitivity-informed run with the stage-1 pruned
#                      ops statically excluded. Identical to stage 1's train.py
#                      invocation in every other respect. Same eval. Does NOT
#                      chain any further.
#
# The two arms share the 32-op snake_case "diff32" vocabulary (train.py:507),
# so the op names mRMR prints feed straight into --pruned-augmentations for the
# `ours` run and are live, not inert (train.py's _active_pruning_vocab).
#
# Usage (submit stage 1; stage 2 is submitted for you):
#   sbatch -J grad_corr_pspnet_cityscapes \
#       job_scripts/train_mrmr_prune_retrain.sbatch pspnet cityscapes
#
#   $1 = backbone   (default: segformer)
#   $2 = dataset    (default: ade20k)
#   $3 = stage-1 exp name (default: grad_corr_${NAME}_${BACKBONE}_${DATASET}_4gpu[_${RUN_TAG}])
#
# Env knobs:
#   STAGE=1|2            which leg to run (default 1; STAGE=2 is set by the
#                        auto-submitted dependent job -- you rarely set it by hand)
#   DOWNWEIGHT=mRMR      stage-1 --corr-downweight-method (default mRMR)
#   CORR_LAMBDA=0.25     stage-1 --corr-lambda (default 0.25; 0.0 is the no-op)
#   NO_STAGE2=1          run stage 1 only, do not submit the retrain
#   PRUNE_MODE=last|union   scrape the LAST 'mRMR pruned' line (default) or the
#                        UNION of every firing round's pruned set
#   REQUIRE_PRUNE=1      abort (don't submit stage 2) if no pruned ops were found
#   RUN_TAG=seed2        suffix exp/job names for repeat runs
#   WORK_DIR=./experiments   parent of experiments/<exp>/
#   CONFIG=configs/nexus.yaml
#   ALLOW_RESUME=1  SKIP_EVAL=1  CONDA_ENV=sensaug  REPO_DIR=/path
#   EXTRA_SBATCH_ARGS="--partition=tron --time=23:59:00"   forwarded to the
#                        stage-2 `sbatch` too
# ==========================================================================

# ---------------------------------------------------------------- config ----
# All overridable via env vars.
NAME="${NAME:-mrmrprune}"

STAGE="${STAGE:-1}"
if [ "${STAGE}" != "1" ] && [ "${STAGE}" != "2" ]; then
    echo "ERROR: STAGE='${STAGE}' must be 1 or 2." >&2
    exit 1
fi

BACKBONE="${1:-segformer}"
DATASET="${2:-ade20k}"

WORK_DIR="${WORK_DIR:-./experiments}"
# gamma is a UMD Nexus partition/account -- use the Nexus cluster config, not della's.
CONFIG="${CONFIG:-configs/nexus.yaml}"
# Your own checkout -- set REPO_DIR to override if it differs from this.
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"
# Canonical path to THIS script for the stage-2 resubmission. Not $BASH_SOURCE:
# under SLURM that points at the spooled copy in /var/spool/slurmd, not the repo.
SELF_PATH="${SELF_PATH:-${REPO_DIR}/job_scripts/train_mrmr_prune_retrain.sbatch}"

# --------------------------------------------------------- per-stage setup ----
EXTRA_ARGS=()

if [ "${STAGE}" = "1" ]; then
    AUG_TYPE="grad_corr"
    EXP_NAME="${3:-grad_corr_${NAME}_${BACKBONE}_${DATASET}_4gpu${RUN_TAG:+_${RUN_TAG}}}"

    # Which function turns the redundancy score into the reweighted training pdf,
    # and how hard. train.py REQUIRES --corr-downweight-method on every grad_corr
    # run (no default), and --corr-lambda defaults to the 0.0 short-circuit, so
    # both must be passed explicitly or the arm silently no-ops. 0.25 is the
    # calibrated value from CLAUDE.md / scripts/calibrate_lambda.py (~80% of ops
    # survive the prune under mRMR).
    DOWNWEIGHT="${DOWNWEIGHT:-mRMR}"
    CORR_LAMBDA="${CORR_LAMBDA:-0.25}"
    EXTRA_ARGS+=(--corr-downweight-method="${DOWNWEIGHT}" --corr-lambda="${CORR_LAMBDA}")

elif [ "${STAGE}" = "2" ]; then
    AUG_TYPE="ours"
    : "${STAGE1_EXP:?STAGE=2 requires STAGE1_EXP in the environment}"
    # "_ours" is already in the name, so train.py won't re-append it and the
    # resume guard below stays in sync with the real work_dir.
    EXP_NAME="${STAGE1_EXP}_pruned_ours"

    # The pruned op list is one space-separated line written by stage 1.
    if [ -n "${PRUNED_AUGS_FILE:-}" ] && [ -s "${PRUNED_AUGS_FILE}" ]; then
        read -r -a PRUNED_OPS < "${PRUNED_AUGS_FILE}"
    else
        PRUNED_OPS=()
    fi
    if [ "${#PRUNED_OPS[@]}" -gt 0 ]; then
        # Unquoted-style expansion, one argv entry per op, for train.py's
        # --pruned-augmentations (nargs="+").
        EXTRA_ARGS+=(--pruned-augmentations "${PRUNED_OPS[@]}")
        echo "stage 2: excluding ${#PRUNED_OPS[@]} mRMR-pruned op(s): ${PRUNED_OPS[*]}"
    else
        echo "NOTE: stage 2 running --aug-type=ours with NO --pruned-augmentations" \
             "(empty or missing PRUNED_AUGS_FILE=${PRUNED_AUGS_FILE:-<unset>})." >&2
    fi
fi

# ---------------------------------------------------------- log grouping ----
# Group this job's bootstrap logs (see the #SBATCH --output/--error comment
# above) into job_scripts/logs/<job-name>/ before doing anything else, so a
# crash further down still leaves clean, grouped logs rather than flat ones.
LOG_BASE="/gammascratch/amodak/SensAug/job_scripts/logs"
LOG_GROUP_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-${NAME}}"
mkdir -p "${LOG_GROUP_DIR}"
for ext in out err; do
    BOOT="${LOG_BASE}/${SLURM_JOB_NAME:-${NAME}}.${SLURM_JOB_ID}.${ext}"
    [ -f "${BOOT}" ] && mv "${BOOT}" "${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-${NAME}}.${SLURM_JOB_ID}.${ext}"
done

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
    echo "Pass a fresh exp name (3rd arg) or RUN_TAG=..., or set ALLOW_RESUME=1 to continue it." >&2
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

echo "=== STAGE ${STAGE}: --aug-type=${AUG_TYPE}  exp=${EXP_NAME} ==="

# ---------------------------------------------------------------- train ----
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
    echo "ERROR: training exited ${TRAIN_STATUS}; skipping evaluation and stage 2." >&2
    exit $TRAIN_STATUS
fi

EXP_DIR="${WORK_DIR}/${EXP_NAME}"

# ------------------------------------------ stage 1 -> stage 2 handoff ----
# Runs before stage 1's own eval so the dependent job is queued early and an
# eval hiccup can't strand it. afterok still gates it on this job exiting 0.
if [ "${STAGE}" = "1" ]; then
    PRUNED_AUGS_FILE="${EXP_DIR}/mrmr_pruned_ops.txt"

    # mRMR prints one line per firing round via print_log(logger="current"):
    #   [redundancy] mRMR pruned N/M ops at lambda=0.25 (budget B of K measured): a, b, c
    # -> the run's mmengine .log (rank 0) and stdout. The prune is re-derived
    # every round, not latched; with --corr-lambda-ramp linear (the default) the
    # final round is the most aggressive, so the LAST line is the run's final
    # decision. PRUNE_MODE=union takes every round's ops instead.
    LOG="$(ls -t "${EXP_DIR}"/*/*.log 2>/dev/null | head -1)"
    : "${LOG:=${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-${NAME}}.${SLURM_JOB_ID}.out}"
    echo "scraping mRMR pruned ops from ${LOG}"

    if [ "${PRUNE_MODE:-last}" = "union" ]; then
        PRUNED_OPS_STR="$(grep -h '\[redundancy\] mRMR pruned' "${LOG}" 2>/dev/null \
            | sed -E 's/.*\): //' | tr ',' '\n' | tr -d '[:blank:]' \
            | grep -v '^$' | sort -u | tr '\n' ' ' | sed 's/ *$//' || true)"
    else
        _line="$(grep -h '\[redundancy\] mRMR pruned' "${LOG}" 2>/dev/null | tail -1 || true)"
        if [ -n "${_line}" ]; then
            PRUNED_OPS_STR="$(printf '%s' "${_line#*): }" | tr ',' ' ' | xargs || true)"
        else
            PRUNED_OPS_STR=""
        fi
    fi

    mkdir -p "${EXP_DIR}"
    printf '%s\n' "${PRUNED_OPS_STR}" > "${PRUNED_AUGS_FILE}"

    if [ -z "${PRUNED_OPS_STR// /}" ]; then
        echo "WARNING: no 'mRMR pruned' ops found in ${LOG}." >&2
        if [ -n "${REQUIRE_PRUNE}" ]; then
            echo "REQUIRE_PRUNE set; not submitting stage 2." >&2
            exit 1
        fi
        echo "stage 2 will run --aug-type=ours with no extra exclusions." >&2
    else
        echo "stage 1 pruned ops -> ${PRUNED_AUGS_FILE}: ${PRUNED_OPS_STR}"
    fi

    if [ -z "${NO_STAGE2}" ]; then
        STAGE2_EXP="${EXP_NAME}_pruned_ours"
        EXTRA_SB=()
        [ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_SB <<< "${EXTRA_SBATCH_ARGS}"
        echo "submitting stage 2 (--dependency=afterok:${SLURM_JOB_ID}) -> ${STAGE2_EXP}"
        sbatch --dependency="afterok:${SLURM_JOB_ID}" \
            -J "${STAGE2_EXP}" \
            "${EXTRA_SB[@]}" \
            --export="ALL,STAGE=2,STAGE1_EXP=${EXP_NAME},PRUNED_AUGS_FILE=${PRUNED_AUGS_FILE}" \
            "${SELF_PATH}" "${BACKBONE}" "${DATASET}"
    else
        echo "NO_STAGE2 set; stage 1 only."
    fi
fi

if [ -n "${SKIP_EVAL}" ]; then
    echo "SKIP_EVAL set; training done, not evaluating."
    exit 0
fi

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

echo "All done (stage ${STAGE}): ${EXP_DIR}"
