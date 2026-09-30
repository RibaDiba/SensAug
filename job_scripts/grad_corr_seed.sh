#!/bin/bash
#
# grad_corr_seed.sh -- submit N seeded repeats of the two-stage mRMR chain across
# every trainable dataset, on one pinned GPU model.
#
# For each (seed x dataset) it submits ONE job: stage 1, the grad_corr + mRMR
# measurement run. That job submits its own stage 2 (the `ours` retrain with the
# ops mRMR pruned) as a SLURM dependent job when it finishes, so only n_seeds x
# datasets jobs are ever queued at once -- the throttle is built in. All the real
# work lives in job_scripts/grad_corr_seed.sbatch.
#
# THERE IS NO CONTROL ARM. This sweep is purely: seeded grad_corr runs, each
# followed by its own pruned retrain. Total jobs = n_seeds x datasets x 2, every
# one of them a leg of that chain. No unpruned `ours` baseline, no
# none/default/random arm, no --no-corr-sa, no --corr-lambda=0, no null
# random-prune arm -- those controls were run separately and are not repeated
# here. Hence, unlike run_all_datasets.sh, this script takes NO aug-type
# argument: the arm is fixed by the stage.
#
# Usage:
#   job_scripts/grad_corr_seed.sh [flags] <n_seeds> <run_iteration> [backbone] [gpu_type]
#
#   <n_seeds>        how many repeats to submit; seeds are numbered 1..n_seeds
#   <run_iteration>  how many times you have run this whole script -- goes into
#                    every experiment name, so relaunching a later generation
#                    never collides with an earlier one
#   [backbone]       default: segformer
#   [gpu_type]       default: rtxa6000. Pinned across BOTH stages of every seed.
#                    On gamma today: rtxa6000 (4/node), rtxa5000 (8/node),
#                    l40s (4/node), rtxa4000 (8/node)
#
#   --dry-run        print the sbatch commands, submit nothing (also DRY_RUN=1)
#   --yes            skip the confirmation prompt (also YES=1)
#   --skip-running   skip a combo whose job name is already in squeue (SKIP_RUNNING=1)
#   --allow-resume   do not skip combos that already hold a checkpoint (see below)
#
# Experiment names, for every seed s and dataset d:
#     base   = seeded_<s>_<d>_<backbone>-<run_iteration>
#     stage1 = <base>_gradcorr        (submitted here)
#     stage2 = <base>_pruned_ours     (submitted by stage 1, --dependency=afterok)
# The SLURM job name matches the experiment name exactly, so
# job_scripts/logs/<job-name>/ and experiments/<exp>/ always agree.
#
# WHAT "SEED" MEANS: a label, not a seeded RNG. train.py hardcodes
# cfg.randomness = dict(seed=0) and set_manual_seed(0), with no flag to vary
# either, so repeats differ only by cuDNN/DDP nondeterminism. See the long note
# at the top of grad_corr_seed.sbatch. Pinning gpu_type is what stops hardware
# from being confounded with that spread -- keep it the same across a generation.
#
# Env knobs (all optional; reach the job through sbatch's --export=ALL default):
#   DATASETS="a b c"      override the dataset list (default: the trainable four)
#   N_GPUS=4              GPUs per job
#   CONFIG=configs/nexus.yaml   cluster config the existence check reads
#   DATA_ROOT=/path/to/data     override data_root from the config
#   SKIP_DATA_CHECK=1     submit every dataset without checking it exists on disk
#   WORK_DIR=./experiments      parent dir for experiments/<exp>/
#   DOWNWEIGHT=mRMR       stage-1 --corr-downweight-method (sbatch default: mRMR)
#   CORR_LAMBDA=0.25      stage-1 --corr-lambda (sbatch default: 0.25)
#   NO_STAGE2=1           run stage 1 only, do not chain the retrain
#   PRUNE_MODE=last|union how stage 1 scrapes its pruned set
#   REQUIRE_PRUNE=1       abort the chain if stage 1 pruned nothing
#   SKIP_EVAL=1           train only, skip test_robust.py / test.py
#   EXTRA_SBATCH_ARGS="--time=36:00:00 --partition=tron"   extra sbatch flags
#   SUBMIT_DELAY=0.3      seconds to sleep between submissions
#
# Cost, from measured sacct walltimes (segformer, 4x rtxa6000, per stage):
#   pascal_voc12 ~1h54m | loveda ~5h10m | ade20k ~6h31m | cityscapes ~7h31m
# One seed is ~21h of stage-1 wall time across 4 concurrent jobs, doubled for
# stage 2. gamma has ~8 rtxa6000:4 nodes total and they are shared, so n_seeds=3
# (12 queued stage-1 jobs) is already most of the partition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/grad_corr_seed.sbatch}"
CONFIG="${CONFIG:-configs/nexus.yaml}"
WORK_DIR="${WORK_DIR:-./experiments}"
N_GPUS="${N_GPUS:-4}"
SUBMIT_DELAY="${SUBMIT_DELAY:-0.3}"

# Exported, not passed as a command prefix, so they reach the job through
# sbatch's --export=ALL default AND so the --dry-run output is a faithful copy of
# what actually gets run. GPU_TYPE is exported after arg parsing, below.
export N_GPUS WORK_DIR

# Datasets that can actually be TRAINED on, not merely present on disk.
#
# Deliberately excludes two that pass a folder-exists check and then die:
#   * acdc -- reuses the backbone's *cityscapes* config and so looks for
#             data/acdc/leftImg8bit/val, but acdc ships rgb_anno/ + gt/. Observed
#             as FileNotFoundError within a minute of start
#             (job_scripts/logs/grad-corr-acdc-2/*.err). Test-only.
#   * idd  -- has no sensaug/custom_configs/mmseg/_base_/datasets/idd.py, so
#             train.py's build_config() raises. Test-only.
# potsdam, synapse and a2i2haze are trainable in principle but are not installed;
# add them here once scripts/prepare_datasets.py has staged them, and the disk
# check below will pick them up automatically.
TRAINABLE_DATASETS=(cityscapes ade20k pascal_voc12 loveda)

VALID_BACKBONES=(pspnet segformer convnext deeplabv3plus swin mae vit)

usage() {
    # print only the leading comment block (line 2 through the first non-# line)
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    exit "${1:-0}"
}

# --- cluster-config parsing (no yaml dep; the file is a flat, unquoted map) ----
# Ported from run_all_datasets.sh rather than sourced, to keep this standalone.

# Value of a top-level `key: value` line.
_config_scalar() {
    awk -v k="$1" '
        $0 ~ "^" k ":[[:space:]]*" {
            sub("^" k ":[[:space:]]*", ""); sub(/[[:space:]]*(#.*)?$/, ""); print; exit
        }' "${CONFIG}"
}

# Sub-path mapped to a dataset key inside the `datasets:` block ("" if absent).
_config_dataset_subpath() {
    awk -v key="$1" '
        /^datasets:[[:space:]]*$/ { inblk=1; next }
        inblk && /^[^[:space:]#]/ { inblk=0 }
        inblk {
            line=$0
            sub(/^[[:space:]]+/, "", line); sub(/[[:space:]]*(#.*)?$/, "", line)
            if (line ~ "^" key ":[[:space:]]*") {
                sub("^" key ":[[:space:]]*", "", line); print line; exit
            }
        }' "${CONFIG}"
}

# 0 if the dataset's folder is present and non-empty (or checks are disabled).
dataset_present() {
    local ds="$1" root sub dir
    [ -n "${SKIP_DATA_CHECK:-}" ] && return 0
    root="${DATA_ROOT:-$(_config_scalar data_root)}"
    if [ -z "${root}" ]; then
        echo "WARNING: could not read data_root from ${CONFIG}; skipping existence checks" >&2
        SKIP_DATA_CHECK=1
        return 0
    fi
    sub="$(_config_dataset_subpath "${ds}")"
    if [ -z "${sub}" ]; then
        echo "WARNING: dataset '${ds}' has no entry under datasets: in ${CONFIG}; skipping it" >&2
        return 1
    fi
    dir="${root}/${sub}"
    if [ ! -d "${dir}" ] || [ -z "$(ls -A "${dir}" 2>/dev/null)" ]; then
        echo "WARNING: dataset '${ds}' not found or empty at ${dir}; skipping it" >&2
        return 1
    fi
    return 0
}

# 0 if a job with this exact name is already queued or running for the user.
job_in_queue() {
    local name="$1"
    squeue -u "${USER:-$(id -un)}" -h -o '%j' 2>/dev/null \
        | sed 's/[[:space:]]*$//' | grep -qxF "${name}"
}

# --------------------------------------------------------------- args ----
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run)       DRY_RUN=1 ;;
        --yes|-y)        YES=1 ;;
        --skip-running)  SKIP_RUNNING=1 ;;
        --allow-resume)  ALLOW_RESUME=1 ;;
        -h|--help|help)  usage 0 ;;
        --*)             echo "ERROR: unknown flag: ${arg}" >&2; usage 1 ;;
        *)               POSITIONAL+=("${arg}") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

[ $# -ge 2 ] || { echo "ERROR: need <n_seeds> and <run_iteration>" >&2; usage 1; }

N_SEEDS="$1"
RUN_ITER="$2"
BACKBONE="${3:-segformer}"
GPU_TYPE="${4:-rtxa6000}"
export GPU_TYPE

case "${N_SEEDS}" in
    ''|*[!0-9]*|0) echo "ERROR: n_seeds='${N_SEEDS}' must be a positive integer." >&2; exit 1 ;;
esac
case "${RUN_ITER}" in
    ''|*[!0-9]*) echo "ERROR: run_iteration='${RUN_ITER}' must be a non-negative integer." >&2; exit 1 ;;
esac

ok=0
for b in "${VALID_BACKBONES[@]}"; do
    [ "${BACKBONE}" = "${b}" ] && ok=1 && break
done
if [ "${ok}" -ne 1 ]; then
    echo "ERROR: backbone '${BACKBONE}' is not supported." >&2
    echo "Valid: ${VALID_BACKBONES[*]}" >&2
    exit 1
fi

if [ -n "${SKIP_RUNNING:-}" ] && ! command -v squeue >/dev/null 2>&1; then
    echo "WARNING: --skip-running set but squeue not found; not checking the queue" >&2
    SKIP_RUNNING=
fi

if [ -n "${DATASETS:-}" ]; then
    read -r -a DATASET_LIST <<< "${DATASETS}"
else
    DATASET_LIST=("${TRAINABLE_DATASETS[@]}")
fi

EXTRA_ARGS=()
[ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_ARGS <<< "${EXTRA_SBATCH_ARGS}"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: sbatch script not found: ${SBATCH_SCRIPT}" >&2
    exit 1
fi

cd "${REPO_DIR}"

if [ -z "${SKIP_DATA_CHECK:-}" ] && [ ! -f "${CONFIG}" ]; then
    echo "WARNING: cluster config '${CONFIG}' not found under ${REPO_DIR}; skipping existence checks" >&2
    SKIP_DATA_CHECK=1
fi

# Pre-filter the dataset list once so a missing dataset is warned about a single
# time, not once per seed.
PRESENT_DATASETS=()
SKIPPED_DATASETS=()
for ds in "${DATASET_LIST[@]}"; do
    if dataset_present "${ds}"; then
        PRESENT_DATASETS+=("${ds}")
    else
        SKIPPED_DATASETS+=("${ds}")
    fi
done

if [ ${#SKIPPED_DATASETS[@]} -gt 0 ]; then
    echo "skipping ${#SKIPPED_DATASETS[@]} unavailable dataset(s): ${SKIPPED_DATASETS[*]}" >&2
fi
if [ ${#PRESENT_DATASETS[@]} -eq 0 ]; then
    echo "ERROR: none of the requested datasets are available; nothing to submit." >&2
    exit 1
fi
DATASET_LIST=("${PRESENT_DATASETS[@]}")

# --------------------------------------------------------------- plan ----
TOTAL=$(( N_SEEDS * ${#DATASET_LIST[@]} ))

echo "Seed sweep plan (run_iteration=${RUN_ITER}):"
echo "  seeds     (${N_SEEDS}): $(seq -s' ' 1 "${N_SEEDS}")"
echo "  datasets  (${#DATASET_LIST[@]}): ${DATASET_LIST[*]}"
echo "  backbone: ${BACKBONE}    gpu: ${GPU_TYPE}:${N_GPUS}"
echo "  stage-1 jobs to submit: ${TOTAL}"
echo "  each chains one stage-2 retrain on success -> up to $(( TOTAL * 2 )) jobs total"

if [ -z "${DRY_RUN:-}" ] && [ -z "${YES:-}" ]; then
    read -r -p "Submit ${TOTAL} stage-1 jobs? [y/N] " reply
    case "${reply}" in
        y|Y|yes|YES) ;;
        *) echo "aborted."; exit 0;;
    esac
fi

# ------------------------------------------------------------- submit ----
count=0
running_skips=0
done_skips=0
for seed in $(seq 1 "${N_SEEDS}"); do
    for ds in "${DATASET_LIST[@]}"; do
        base="seeded_${seed}_${ds}_${BACKBONE}-${RUN_ITER}"
        exp="${base}_gradcorr"

        if [ -n "${SKIP_RUNNING:-}" ] && job_in_queue "${exp}"; then
            echo "skip: '${exp}' is already in the queue" >&2
            running_skips=$((running_skips + 1))
            continue
        fi
        # Don't resubmit a combo that already finished: the sbatch would refuse
        # it anyway (its ALLOW_RESUME guard), but only after burning an
        # allocation and mailing a FAIL. Catch it here for free.
        if [ -z "${ALLOW_RESUME:-}" ] && [ -f "${WORK_DIR}/${exp}/last_checkpoint" ]; then
            echo "skip: '${exp}' already holds a checkpoint (--allow-resume to override)" >&2
            done_skips=$((done_skips + 1))
            continue
        fi

        cmd=(sbatch -J "${exp}"
             --gres="gpu:${GPU_TYPE}:${N_GPUS}"
             "${EXTRA_ARGS[@]}"
             "${SBATCH_SCRIPT}" "${seed}" "${RUN_ITER}" "${ds}" "${BACKBONE}")
        if [ -n "${DRY_RUN:-}" ]; then
            printf '%s\n' "${cmd[*]}"
        else
            echo "submit: seed=${seed} dataset=${ds} -> ${exp}"
            "${cmd[@]}"
            sleep "${SUBMIT_DELAY}"
        fi
        count=$((count + 1))
    done
done

[ "${running_skips}" -gt 0 ] && echo "skipped ${running_skips} combo(s) already in the queue" >&2
[ "${done_skips}" -gt 0 ] && echo "skipped ${done_skips} combo(s) that already hold a checkpoint" >&2

if [ -n "${DRY_RUN:-}" ]; then
    echo "# DRY_RUN: ${count} stage-1 jobs would be submitted" \
         "(backbone=${BACKBONE}, gpu=${GPU_TYPE}:${N_GPUS}, run_iteration=${RUN_ITER})"
else
    echo "${count} stage-1 jobs submitted" \
         "(backbone=${BACKBONE}, gpu=${GPU_TYPE}:${N_GPUS}, run_iteration=${RUN_ITER});" \
         "each chains its own stage-2 retrain on success."
fi
