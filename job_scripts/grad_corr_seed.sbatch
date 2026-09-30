#!/bin/bash

#SBATCH --job-name=grad-corr-seed
#SBATCH --nodes=1

## GAMMA training config (per the lab's train_generic.sh guidelines)
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
## Overridden on the sbatch command line by grad_corr_seed.sh (--gres=gpu:$GPU_TYPE:$N_GPUS),
## and re-passed explicitly when this script submits its own stage 2, so every leg of a
## seed's chain lands on the SAME GPU model. That is the whole point of pinning it: with
## the training seed fixed at 0 (see the header note below), hardware is the one remaining
## variable that could masquerade as run-to-run spread.
#SBATCH --gres=gpu:rtxa6000:4
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
# grad_corr_seed.sbatch -- one seed's two-stage chain for ONE dataset.
#
#   STAGE 1 (default)  --aug-type=grad_corr  --corr-downweight-method=mRMR
#                      Measures the gradient cross-correlation matrix R and lets
#                      mRMR hard-prune the training pdf. Afterwards it scrapes the
#                      final set of ops mRMR drove to probability 0 out of the run's
#                      mmengine log, writes them to
#                      <work_dir>/<exp>/mrmr_pruned_ops.txt, and submits STAGE 2 as
#                      a SLURM dependent job (--dependency=afterok).
#
#   STAGE 2            --aug-type=ours  --pruned-augmentations <ops from stage 1>
#                      A sensitivity-informed run with stage 1's pruned ops
#                      statically excluded. Identical to stage 1's train.py
#                      invocation in every other respect. Does NOT chain further.
#
# THERE IS NO CONTROL ARM HERE, BY DESIGN. This script runs exactly those two
# legs. No unpruned `ours` baseline, no `none`/`default`/`random`, no
# --no-corr-sa, no --corr-lambda=0, no --random-prune-method=null. Those controls
# were run separately and are not re-run by the seed sweep. (Each grad_corr run
# does still emit its round-3 "control probe" into its own corr_matrix_log.json
# with "role": "control" -- that is a measurement *inside* the run, governed by
# configs/rounds.yaml, not a separate job.)
#
# Both arms share the 32-op snake_case "diff32" vocabulary, so the op names mRMR
# prints feed straight into --pruned-augmentations for the `ours` run and are
# live, not inert (see train.py's _active_pruning_vocab).
#
# WHAT "SEED" MEANS HERE: it is a LABEL, not a seeded RNG. train.py hardcodes
# cfg.randomness = dict(seed=0) (train.py:1099) and set_manual_seed(0)
# (train.py:1121); no flag, env var or config varies either, and the augmentation
# sampler draws from the global numpy stream that descends from them. Repeats of
# the same (dataset, backbone) therefore differ only by cuDNN/DDP nondeterminism
# -- default_runtime.py sets cudnn_benchmark=True and nothing sets
# deterministic=True. That spread is real and worth measuring, but it is
# run-to-run noise, NOT a seed sweep; describe it that way. Pinning GPU_TYPE is
# what keeps hardware from being confounded with it.
#
# Usage -- normally you do not call this directly; use:
#     job_scripts/grad_corr_seed.sh <n_seeds> <run_iteration> [backbone] [gpu_type]
#   To submit a single combo by hand (stage 2 is submitted for you):
#     sbatch -J seeded_1_loveda_segformer-1_gradcorr \
#         job_scripts/grad_corr_seed.sbatch 1 1 loveda segformer
#
#   $1 = seed number      (default: 1)   -- a label; see above
#   $2 = run_iteration    (default: 1)   -- which run of the whole sweep this is
#   $3 = dataset          (default: loveda)
#   $4 = backbone         (default: segformer)
#
# Experiment names are derived, never passed:
#     base   = seeded_<seed>_<dataset>_<backbone>-<run_iteration>
#     stage1 = <base>_gradcorr
#     stage2 = <base>_pruned_ours
# The suffixes are baked in rather than left to train.py, which would append the
# same "_gradcorr"/"_ours" itself (train.py:1637-1646). Its guards are idempotent
# (`if suffix not in exp_name`), so spelling them here changes nothing about the
# run and makes the final directory name predictable BEFORE submission -- which
# is what lets -J, the log group dir and the work_dir all agree.
#
# Env knobs:
#   STAGE=1|2            which leg to run (default 1; STAGE=2 is set by the
#                        auto-submitted dependent job -- you rarely set it by hand)
#   GPU_TYPE=rtxa6000    GPU model, re-passed to stage 2's sbatch
#   N_GPUS=4             GPUs per job, ditto
#   DOWNWEIGHT=mRMR      stage-1 --corr-downweight-method (default mRMR)
#   CORR_LAMBDA=0.25     stage-1 --corr-lambda (default 0.25; 0.0 is the no-op)
#   NO_STAGE2=1          run stage 1 only, do not submit the retrain
#   PRUNE_MODE=last|union   scrape the LAST 'mRMR pruned' line (default) or the
#                        UNION of every firing round's pruned set
#   REQUIRE_PRUNE=1      abort (don't submit stage 2) if no pruned ops were found
#   WORK_DIR=./experiments   parent of experiments/<exp>/
#   CONFIG=configs/nexus.yaml
#   ALLOW_RESUME=1  SKIP_EVAL=1  CONDA_ENV=sensaug  REPO_DIR=/path
#   EXTRA_SBATCH_ARGS="--partition=tron --time=23:59:00"   forwarded to stage 2 too
#
# NOTE ON --pruned-augmentations: this script deliberately declares NO default
# prune list, unlike train_nexus_gamma.sbatch (whose PRUNED_AUGS defaults to five
# ops and -- because ${VAR:-default} substitutes on empty as well as unset --
# cannot be emptied from the environment). Stage 1 must measure R over the full
# 30-op bank (32 minus --no-inv-aug's two) or mRMR would be ranking a bank that
# had already been cut, and stage 2's prune would inherit that.
#
# KNOWN ISSUE, not fixed here: the post-training eval leg is the part that failed
# the last time a sweep ran. On ade20k, test_robust.py:257 raised
#   IndexError: The shape of the mask [512, 683] does not match the shape of the
#   indexed tensor [300, 400]
# -- the image/label frame mismatch that _matrix_in_frame fixed on the TRAINING
# path but not in the robustness eval. Training completes first and the chain to
# stage 2 is submitted before the eval runs, so an eval failure cannot strand the
# retrain. Set SKIP_EVAL=1 to skip it entirely.
# ==========================================================================

# ---------------------------------------------------------------- config ----
STAGE="${STAGE:-1}"
if [ "${STAGE}" != "1" ] && [ "${STAGE}" != "2" ]; then
    echo "ERROR: STAGE='${STAGE}' must be 1 or 2." >&2
    exit 1
fi

# `${N-default}`, not `${N:-default}`: the latter also substitutes on an EMPTY
# argument, so `sbatch ... "" 1 loveda segformer` would silently become seed 1 and
# resume -- or collide with -- a real seed-1 run. Defaulting only when the arg is
# absent lets the validation below reject an empty one instead.
SEED="${1-1}"
RUN_ITER="${2-1}"
DATASET="${3-loveda}"
BACKBONE="${4-segformer}"

# Validate here rather than letting these flow into a directory name --
# "seeded_foo_loveda_segformer-1" would be accepted by everything downstream and
# silently produce an unparseable experiment name.
for _pair in "SEED:${SEED}" "RUN_ITER:${RUN_ITER}"; do
    _name="${_pair%%:*}"; _val="${_pair#*:}"
    case "${_val}" in
        ''|*[!0-9]*) echo "ERROR: ${_name}='${_val}' must be a non-negative integer." >&2; exit 1 ;;
    esac
done
for _pair in "DATASET:${DATASET}" "BACKBONE:${BACKBONE}"; do
    _name="${_pair%%:*}"; _val="${_pair#*:}"
    case "${_val}" in
        ''|*[!a-zA-Z0-9_]*) echo "ERROR: ${_name}='${_val}' must be a non-empty [A-Za-z0-9_] name." >&2; exit 1 ;;
    esac
done

GPU_TYPE="${GPU_TYPE:-rtxa6000}"
N_GPUS="${N_GPUS:-4}"

WORK_DIR="${WORK_DIR:-./experiments}"
# gamma is a UMD Nexus partition/account -- use the Nexus cluster config, not della's.
CONFIG="${CONFIG:-configs/nexus.yaml}"
# Your own checkout -- set REPO_DIR to override if it differs from this.
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"
# Canonical path to THIS script for the stage-2 resubmission. Not $BASH_SOURCE:
# under SLURM that points at the spooled copy in /var/spool/slurmd, not the repo.
SELF_PATH="${SELF_PATH:-${REPO_DIR}/job_scripts/grad_corr_seed.sbatch}"

# ------------------------------------------------------------- exp names ----
BASE_NAME="seeded_${SEED}_${DATASET}_${BACKBONE}-${RUN_ITER}"
STAGE1_EXP="${BASE_NAME}_gradcorr"
STAGE2_EXP="${BASE_NAME}_pruned_ours"

# --------------------------------------------------------- per-stage setup ----
EXTRA_ARGS=()

if [ "${STAGE}" = "1" ]; then
    AUG_TYPE="grad_corr"
    EXP_NAME="${STAGE1_EXP}"

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
    EXP_NAME="${STAGE2_EXP}"

    # The pruned op list is one space-separated line written by stage 1. Its path
    # is passed through --export, but recompute a default so a hand-run stage 2
    # still finds it.
    PRUNED_AUGS_FILE="${PRUNED_AUGS_FILE:-${WORK_DIR}/${STAGE1_EXP}/mrmr_pruned_ops.txt}"
    if [ -s "${PRUNED_AUGS_FILE}" ]; then
        read -r -a PRUNED_OPS < "${PRUNED_AUGS_FILE}"
    else
        PRUNED_OPS=()
    fi
    if [ "${#PRUNED_OPS[@]}" -gt 0 ]; then
        EXTRA_ARGS+=(--pruned-augmentations "${PRUNED_OPS[@]}")
        echo "stage 2: excluding ${#PRUNED_OPS[@]} mRMR-pruned op(s): ${PRUNED_OPS[*]}"
    else
        echo "NOTE: stage 2 running --aug-type=ours with NO --pruned-augmentations" \
             "(empty or missing PRUNED_AUGS_FILE=${PRUNED_AUGS_FILE})." >&2
    fi
fi

# ---------------------------------------------------------- log grouping ----
# Group this job's bootstrap logs (see the #SBATCH --output/--error comment
# above) into job_scripts/logs/<job-name>/ before doing anything else, so a
# crash further down still leaves clean, grouped logs rather than flat ones.
# Anchored to the same absolute base as the #SBATCH paths, so this works
# regardless of where `sbatch` was actually invoked from.
# Overridable only so tests can point it at a tmpdir; real runs must leave it at
# the absolute default, which is the same path the #SBATCH directives above use.
LOG_BASE="${LOG_BASE:-/gammascratch/amodak/SensAug/job_scripts/logs}"
LOG_GROUP_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-${EXP_NAME}}"
mkdir -p "${LOG_GROUP_DIR}"
for ext in out err; do
    BOOT="${LOG_BASE}/${SLURM_JOB_NAME:-${EXP_NAME}}.${SLURM_JOB_ID}.${ext}"
    [ -f "${BOOT}" ] && mv "${BOOT}" "${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-${EXP_NAME}}.${SLURM_JOB_ID}.${ext}"
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
    echo "Bump the run_iteration (2nd arg), or set ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

# ------------------------------------------------------- environment setup ----
# `module` is a shell FUNCTION defined by /etc/profile.d/modules.sh, not a binary.
# An interactive login shell has it and exports it, so SLURM's `--export=ALL`
# carries it into a job submitted BY HAND and `module load` just works. A job
# submitted from a NON-interactive shell -- a wrapper script, cron, an agent --
# inherits no such function, and then `module load` dies with 127, `conda` dies
# with 127, and `torchrun` dies with 127 one second into the job. That is exactly
# how jobs 7554655/7554656 failed. So re-source it rather than assuming it.
# `${VAR-default}`, so MODULE_INIT_FILES="" explicitly means "source nothing" --
# that is the seam the no-module test uses to exercise this path hermetically,
# since sourcing the real init would let `module load` prepend the real conda
# ahead of the test's stubs and launch actual training.
MODULE_INIT_FILES="${MODULE_INIT_FILES-/etc/profile.d/modules.sh /etc/profile.d/ummodules.sh}"
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    for _init in ${MODULE_INIT_FILES}; do
        # shellcheck disable=SC1090
        [ -f "${_init}" ] && source "${_init}"
    done
fi

# Prefer the module, but fall back to sourcing miniforge's own conda.sh: the
# modulefile only exists on nodes whose MODULEPATH carries it, whereas the install
# path is stable. CONDA_SH overrides both.
module load miniforge3 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
    for _conda_sh in "${CONDA_SH:-}" \
                     /opt/common/miniforge3/latest/etc/profile.d/conda.sh \
                     "${HOME}/miniforge3/etc/profile.d/conda.sh"; do
        if [ -n "${_conda_sh}" ] && [ -f "${_conda_sh}" ]; then
            # shellcheck disable=SC1090
            source "${_conda_sh}" && break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Tried 'module load miniforge3' and" >&2
    echo "       /opt/common/miniforge3/latest/etc/profile.d/conda.sh." >&2
    echo "       Set CONDA_SH=/path/to/etc/profile.d/conda.sh and resubmit." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-sensaug}" || {
    echo "ERROR: could not activate conda env '${CONDA_ENV:-sensaug}'." >&2
    exit 1
}

# torch ships its own CUDA runtime, so a missing module is a warning, not fatal.
module load cuda/11.7.0 2>/dev/null \
    || echo "WARNING: could not 'module load cuda/11.7.0'; relying on torch's bundled runtime." >&2

# Fail HERE with a legible message rather than 30 lines later as a bare exit 127
# out of torchrun, which is what makes this class of failure hard to read.
if ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun not on PATH after activating '${CONDA_ENV:-sensaug}'." >&2
    echo "       PATH=${PATH}" >&2
    exit 1
fi

# Nexus compute nodes have internet access (unlike della's), so no pretrained-checkpoint
# pre-staging is needed here -- see sensaug/cluster_config.py.
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-${N_GPUS}}}
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

echo "=== STAGE ${STAGE}: seed=${SEED} run_iter=${RUN_ITER} dataset=${DATASET}" \
     "backbone=${BACKBONE} gpu=${GPU_TYPE}:${N_GPUS} aug=${AUG_TYPE} exp=${EXP_NAME} ==="

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
    # The sed is greedy on purpose: the real lines carry a nested parenthetical
    # ("... measured, 1 exempt (no usable row in R)): a, b, c"), and matching to
    # the LAST "): " is what clears it.
    LOG="$(ls -t "${EXP_DIR}"/*/*.log 2>/dev/null | head -1)"
    : "${LOG:=${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-${EXP_NAME}}.${SLURM_JOB_ID}.out}"
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
        EXTRA_SB=()
        [ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_SB <<< "${EXTRA_SBATCH_ARGS}"
        echo "submitting stage 2 (--dependency=afterok:${SLURM_JOB_ID}) -> ${STAGE2_EXP}"
        # --gres is re-passed so stage 2 runs on the same GPU model as stage 1;
        # the #SBATCH directive in this file would otherwise pull it back to the
        # rtxa6000 default whenever GPU_TYPE was overridden at submit time.
        sbatch --dependency="afterok:${SLURM_JOB_ID}" \
            -J "${STAGE2_EXP}" \
            --gres="gpu:${GPU_TYPE}:${N_GPUS}" \
            "${EXTRA_SB[@]}" \
            --export="ALL,STAGE=2,PRUNED_AUGS_FILE=${PRUNED_AUGS_FILE}" \
            "${SELF_PATH}" "${SEED}" "${RUN_ITER}" "${DATASET}" "${BACKBONE}"
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
    --use-best \
|| echo "WARNING: test_robust.py failed; continuing (stage 2 was already queued)." >&2

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
