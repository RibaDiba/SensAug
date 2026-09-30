#!/bin/bash
#
# run_all_datasets.sh -- submit one training job per dataset for a SINGLE
# backbone, across both augmentation arms (`ours` and `grad_corr` with mRMR).
#
# Each combination is handed to job_scripts/train_nexus_gamma.sbatch, which does
# the actual 4-GPU torchrun + post-train robustness eval. That script already:
#   * validates AUG_TYPE,
#   * defaults CONFIG=configs/nexus.yaml and WORK_DIR=./experiments,
#   * for AUG_TYPE=grad_corr appends --corr-downweight-method=$DOWNWEIGHT
#     (DOWNWEIGHT defaults to mRMR) and --corr-lambda=$CORR_LAMBDA (default 0.25),
#   * groups logs under job_scripts/logs/<job-name>/ from `sbatch -J <name>`,
#   * refuses to resume an existing experiments/<exp>/ unless ALLOW_RESUME=1.
# So nothing about Lever 3 needs to be passed from here -- picking `grad_corr`
# as the aug type is enough to get the mRMR arm.
#
# Usage:
#   job_scripts/run_all_datasets.sh [--skip-running] <backbone> [aug_type ...]
#
#   <backbone>        one of: pspnet segformer convnext deeplabv3plus swin mae vit
#   [aug_type ...]    optional; defaults to: ours grad_corr
#   --skip-running    skip any combo whose job name is already queued/running in
#                     squeue for $USER (same as SKIP_RUNNING=1)
#   --no-skip-running explicitly disable that check (overrides SKIP_RUNNING)
#
# Before submitting, each dataset is checked for existence on disk: its folder is
# resolved from `data_root` + the `datasets:` mapping in the cluster config
# (CONFIG, default configs/nexus.yaml). A dataset whose folder is missing, empty,
# or absent from the config is WARNED about and skipped -- the sweep continues
# with whatever is installed. Set SKIP_DATA_CHECK=1 to submit regardless.
#
# Env knobs (all optional; exported through to the sbatch job):
#   DATASETS="a b c"       override the dataset list (default: all 9 in nexus.yaml)
#   CONFIG=configs/nexus.yaml   cluster config the existence check reads
#   DATA_ROOT=/path/to/data     override data_root from the config
#   SKIP_DATA_CHECK=1     submit every dataset without checking it exists on disk
#   SKIP_RUNNING=1        skip combos already present in squeue (see --skip-running)
#   RUN_TAG=seed2          suffix job/exp names for repeat runs
#   WORK_DIR=./experiments parent dir for experiments/<exp>/
#   DOWNWEIGHT=mRMR        grad_corr down-weighting method (sbatch default: mRMR)
#   CORR_LAMBDA=0.25       grad_corr Lever 3 budget/strength (sbatch default: 0.25)
#   ALLOW_RESUME=1         let a job resume an existing checkpoint
#   SKIP_EVAL=1            train only, skip test_robust.py / test.py
#   EXTRA_SBATCH_ARGS="--time=24:00:00 --partition=tron"   extra sbatch flags
#   DRY_RUN=1             print the sbatch commands, submit nothing
#   SUBMIT_DELAY=0.3      seconds to sleep between submissions
#
# Dataset caveats (see CLAUDE.md "three augmentation vocabularies" / build_config):
#   * idd  -- has no sensaug/custom_configs/mmseg/_base_/datasets/idd.py, so
#             train.py:build_config() raises. Drop it from DATASETS or add that
#             base config before relying on it.
#   * acdc -- reuses the backbone's *cityscapes* config and the 19-class
#             Cityscapes label space; only resolves for backbones that ship a
#             cityscapes config (pspnet segformer convnext deeplabv3plus vit).
#             Existing scripts treat acdc as test-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/train_nexus_gamma.sbatch}"
CONFIG="${CONFIG:-configs/nexus.yaml}"

# --- cluster-config parsing (no yaml dep; the file is a flat, unquoted map) ----

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

# 0 if a job with this exact name is already queued or running for the user.
job_in_queue() {
    local name="$1"
    squeue -u "${USER:-$(id -un)}" -h -o '%j' 2>/dev/null \
        | sed 's/[[:space:]]*$//' | grep -qxF "${name}"
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

VALID_BACKBONES=(pspnet segformer convnext deeplabv3plus swin mae vit)

DATASETS_DEFAULT=(cityscapes ade20k pascal_voc12 loveda potsdam synapse a2i2haze acdc idd)

usage() {
    # print only the leading comment block (line 2 through the first non-# line)
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    exit "${1:-0}"
}

# Pull flags out of the arg list; everything else stays positional
# (<backbone> then aug types).
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --skip-running)    SKIP_RUNNING=1 ;;
        --no-skip-running) SKIP_RUNNING= ;;
        -h|--help|help)    usage 0 ;;
        --*)               echo "ERROR: unknown flag: ${arg}" >&2; usage 1 ;;
        *)                 POSITIONAL+=("${arg}") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

[ $# -ge 1 ] || { echo "ERROR: missing <backbone>" >&2; usage 1; }

BACKBONE="$1"
shift

if [ -n "${SKIP_RUNNING:-}" ] && ! command -v squeue >/dev/null 2>&1; then
    echo "WARNING: --skip-running set but squeue not found; not checking the queue" >&2
    SKIP_RUNNING=
fi

ok=0
for b in "${VALID_BACKBONES[@]}"; do
    [ "${BACKBONE}" = "${b}" ] && ok=1 && break
done
if [ "${ok}" -ne 1 ]; then
    echo "ERROR: backbone '${BACKBONE}' is not supported." >&2
    echo "Valid: ${VALID_BACKBONES[*]}" >&2
    exit 1
fi

AUG_TYPES=("$@")
[ ${#AUG_TYPES[@]} -gt 0 ] || AUG_TYPES=(ours grad_corr)

if [ -n "${DATASETS:-}" ]; then
    read -r -a DATASET_LIST <<< "${DATASETS}"
else
    DATASET_LIST=("${DATASETS_DEFAULT[@]}")
fi

EXTRA_ARGS=()
[ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_ARGS <<< "${EXTRA_SBATCH_ARGS}"

SUBMIT_DELAY="${SUBMIT_DELAY:-0.3}"

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
# time, not once per aug type.
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

count=0
running_skips=0
for aug in "${AUG_TYPES[@]}"; do
    for ds in "${DATASET_LIST[@]}"; do
        exp="${aug}_${BACKBONE}_${ds}${RUN_TAG:+_${RUN_TAG}}"
        if [ -n "${SKIP_RUNNING:-}" ] && job_in_queue "${exp}"; then
            echo "skip: '${exp}' is already in the queue" >&2
            running_skips=$((running_skips + 1))
            continue
        fi
        cmd=(sbatch -J "${exp}" "${EXTRA_ARGS[@]}" "${SBATCH_SCRIPT}" "${aug}" "${BACKBONE}" "${ds}" "${exp}")
        if [ -n "${DRY_RUN:-}" ]; then
            printf '%s\n' "${cmd[*]}"
        else
            echo "submit: aug=${aug} backbone=${BACKBONE} dataset=${ds} -> ${exp}"
            "${cmd[@]}"
            sleep "${SUBMIT_DELAY}"
        fi
        count=$((count + 1))
    done
done

[ "${running_skips}" -gt 0 ] && echo "skipped ${running_skips} combo(s) already in the queue" >&2

if [ -n "${DRY_RUN:-}" ]; then
    echo "# DRY_RUN: ${count} jobs would be submitted for backbone=${BACKBONE} (augs: ${AUG_TYPES[*]})"
else
    echo "${count} jobs submitted for backbone=${BACKBONE} (augs: ${AUG_TYPES[*]})"
fi
