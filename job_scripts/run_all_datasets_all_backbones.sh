#!/bin/bash
#
# run_all_datasets_all_backbones.sh -- the full sweep: every backbone x every
# dataset x both augmentation arms (`ours` and `grad_corr` with mRMR).
#
# Thin wrapper that calls job_scripts/run_all_datasets.sh once per backbone;
# all the real work (validation, sbatch submission, grad_corr/mRMR wiring) lives
# there and in job_scripts/train_nexus_gamma.sbatch.
#
# Usage:
#   job_scripts/run_all_datasets_all_backbones.sh [--skip-running] [aug_type ...]
#
#   [aug_type ...]    optional; defaults to: ours grad_corr
#   --skip-running    skip any combo whose job name is already queued/running in
#                     squeue for $USER (same as SKIP_RUNNING=1)
#   --no-skip-running explicitly disable that check
#
# run_all_datasets.sh checks each dataset exists on disk (folder from data_root +
# the datasets: map in CONFIG, default configs/nexus.yaml) and skips + warns for
# any that are missing, so the printed total below is an upper bound. Set
# SKIP_DATA_CHECK=1 to submit regardless.
#
# Env knobs (optional; inherited by run_all_datasets.sh -> the sbatch job):
#   BACKBONES="a b"        override the backbone list
#                          (default: pspnet segformer convnext deeplabv3plus swin mae vit)
#   DATASETS="a b c"       override the dataset list (default: all 9 in nexus.yaml)
#   CONFIG=configs/nexus.yaml  DATA_ROOT=/path  SKIP_DATA_CHECK=1   dataset check
#   SKIP_RUNNING=1        skip combos already present in squeue (see --skip-running)
#   RUN_TAG=seed2          suffix job/exp names for repeat runs
#   WORK_DIR=./experiments
#   DOWNWEIGHT=mRMR  CORR_LAMBDA=0.25   grad_corr Lever 3 (sbatch defaults)
#   ALLOW_RESUME=1  SKIP_EVAL=1  EXTRA_SBATCH_ARGS="..."
#   DRY_RUN=1             print the sbatch commands, submit nothing, skip the prompt
#   YES=1                skip the confirmation prompt and submit
#
# At the default geometry this is 7 backbones x 9 datasets x 2 augs = 126 jobs
# (minus any datasets not yet installed). See run_all_datasets.sh for the
# idd / acdc dataset caveats.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHILD="${SCRIPT_DIR}/run_all_datasets.sh"

usage() {
    # print only the leading comment block (line 2 through the first non-# line)
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    exit "${1:-0}"
}
# Pull flags out; the rest are aug types. SKIP_RUNNING is exported so the child
# picks it up without the flag being forwarded.
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --skip-running)    export SKIP_RUNNING=1 ;;
        --no-skip-running) export SKIP_RUNNING= ;;
        -h|--help|help)    usage 0 ;;
        --*)               echo "ERROR: unknown flag: ${arg}" >&2; usage 1 ;;
        *)                 POSITIONAL+=("${arg}") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

[ -x "${CHILD}" ] || [ -f "${CHILD}" ] || { echo "ERROR: ${CHILD} not found" >&2; exit 1; }

BACKBONES_DEFAULT=(pspnet segformer convnext deeplabv3plus swin mae vit)
DATASETS_DEFAULT=(cityscapes ade20k pascal_voc12 loveda potsdam synapse a2i2haze acdc idd)

if [ -n "${BACKBONES:-}" ]; then
    read -r -a BACKBONE_LIST <<< "${BACKBONES}"
else
    BACKBONE_LIST=("${BACKBONES_DEFAULT[@]}")
fi

if [ -n "${DATASETS:-}" ]; then
    read -r -a DATASET_LIST <<< "${DATASETS}"
else
    DATASET_LIST=("${DATASETS_DEFAULT[@]}")
fi

AUG_TYPES=("$@")
[ ${#AUG_TYPES[@]} -gt 0 ] || AUG_TYPES=(ours grad_corr)

TOTAL=$(( ${#BACKBONE_LIST[@]} * ${#DATASET_LIST[@]} * ${#AUG_TYPES[@]} ))

echo "Sweep plan:"
echo "  backbones (${#BACKBONE_LIST[@]}): ${BACKBONE_LIST[*]}"
echo "  datasets  (${#DATASET_LIST[@]}): ${DATASET_LIST[*]}"
echo "  augs      (${#AUG_TYPES[@]}): ${AUG_TYPES[*]}"
echo "  total jobs: up to ${TOTAL} (datasets missing on disk${SKIP_RUNNING:+, and combos already in the queue} are skipped)"

if [ -z "${DRY_RUN:-}" ] && [ -z "${YES:-}" ]; then
    read -r -p "Submit up to ${TOTAL} jobs? [y/N] " reply
    case "${reply}" in
        y|Y|yes|YES) ;;
        *) echo "aborted."; exit 0;;
    esac
fi

for b in "${BACKBONE_LIST[@]}"; do
    echo "=== backbone: ${b} ==="
    bash "${CHILD}" "${b}" "${AUG_TYPES[@]}"
done

echo "Done: swept ${#BACKBONE_LIST[@]} backbone(s) x available datasets x ${#AUG_TYPES[@]} aug(s) (augs: ${AUG_TYPES[*]}); see per-backbone counts above."
