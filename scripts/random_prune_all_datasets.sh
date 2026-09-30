#!/bin/bash
#
# random_prune_all_datasets.sh -- submit the random-pruning control for every
# dataset at once: "ours pruned", but with the pruned ops drawn at random.
#
# One job per entry in PRUNE_COUNTS below. The work lives in the paired
# scripts/random_prune_all_datasets.sbatch; see its header for why this is not
# launched through job_scripts/train_nexus_gamma.sbatch.
#
# Usage:
#   scripts/random_prune_all_datasets.sh [--dry-run] [--yes] [--skip-running] [backbone]
#
#   [backbone]       default: segformer
#   --dry-run        print the sbatch commands, submit nothing (also DRY_RUN=1)
#   --yes            skip the confirmation prompt (also YES=1)
#   --skip-running   skip a dataset whose job name is already in squeue
#
# Job name:  ours-pruned-random-<dataset>
# Exp dir:   experiments/ours-pruned-random-<dataset>_nullprune<N>_s<seed>
#            (train.py appends the _nullprune<N>_s<seed> part to every null-arm
#            run, so two draws can never share a work_dir.)
#
# Env knobs:
#   DATASETS="a b"   restrict to a subset of PRUNE_COUNTS' keys
#   GPU_TYPE=rtxa6000  N_GPUS=4   -- the hardware the pruned runs used; keep it
#   SEED_OFFSET=0    shift every seed, for a second independent round of draws
#   EXTRA_SBATCH_ARGS="--time=36:00:00"
#   SAVE_VIS=1       also write overlay PNGs during eval (off: see the sbatch)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SBATCH_SCRIPT="${SCRIPT_DIR}/random_prune_all_datasets.sbatch"
WORK_DIR="${WORK_DIR:-./experiments}"
GPU_TYPE="${GPU_TYPE:-rtxa6000}"
N_GPUS="${N_GPUS:-4}"
SEED_OFFSET="${SEED_OFFSET:-0}"
export WORK_DIR SAVE_VIS="${SAVE_VIS:-}"

# ---- dataset -> number of ops to drop at random ------------------------------
# N=5 for every dataset: the most common count across the mRMR-pruned runs
# (which pruned 5 or 6 depending on the run). Edit a value to match a specific
# partner run; `wc -w < experiments/<run>/mrmr_pruned_ops.txt` gives its count.
declare -A PRUNE_COUNTS=(
    [ade20k]=5
    [cityscapes]=5
    [loveda]=5
    [pascal_voc12]=5
)

# A distinct seed per dataset, so the four controls are four independent draws
# rather than the same 5 ops pruned everywhere (the eligible pool is identical
# across datasets, so one shared seed would give one shared draw).
declare -A SEEDS=(
    [ade20k]=101
    [cityscapes]=102
    [loveda]=103
    [pascal_voc12]=104
)

usage() {
    sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    exit "${1:-0}"
}

POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run)      DRY_RUN=1 ;;
        --yes|-y)       YES=1 ;;
        --skip-running) SKIP_RUNNING=1 ;;
        -h|--help)      usage 0 ;;
        --*)            echo "ERROR: unknown flag: ${arg}" >&2; usage 1 ;;
        *)              POSITIONAL+=("${arg}") ;;
    esac
done
BACKBONE="${POSITIONAL[0]:-segformer}"

if [ -n "${DATASETS:-}" ]; then
    read -r -a DATASET_LIST <<< "${DATASETS}"
else
    mapfile -t DATASET_LIST < <(printf '%s\n' "${!PRUNE_COUNTS[@]}" | sort)
fi
for ds in "${DATASET_LIST[@]}"; do
    [ -n "${PRUNE_COUNTS[$ds]:-}" ] && [ -n "${SEEDS[$ds]:-}" ] || {
        echo "ERROR: '${ds}' needs an entry in both PRUNE_COUNTS and SEEDS." >&2; exit 1; }
done

EXTRA_ARGS=()
[ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_ARGS <<< "${EXTRA_SBATCH_ARGS}"

cd "${REPO_DIR}"

echo "Random-prune control plan (backbone=${BACKBONE}, gpu=${GPU_TYPE}:${N_GPUS}):"
for ds in "${DATASET_LIST[@]}"; do
    printf '  %-13s drop %s ops  seed %s\n' "${ds}" "${PRUNE_COUNTS[$ds]}" \
        "$(( SEEDS[$ds] + SEED_OFFSET ))"
done

if [ -z "${DRY_RUN:-}" ] && [ -z "${YES:-}" ]; then
    read -r -p "Submit ${#DATASET_LIST[@]} jobs? [y/N] " reply
    case "${reply}" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

count=0
for ds in "${DATASET_LIST[@]}"; do
    n="${PRUNE_COUNTS[$ds]}"
    seed=$(( SEEDS[$ds] + SEED_OFFSET ))
    job="ours-pruned-random-${ds}"
    exp="${job}_nullprune${n}_s${seed}"

    if [ -n "${SKIP_RUNNING:-}" ] && squeue -u "${USER}" -h -o '%j' 2>/dev/null \
            | grep -qxF "${job}"; then
        echo "skip: ${job} is already in the queue" >&2
        continue
    fi
    # The sbatch would refuse this anyway, but only after taking an allocation.
    if [ -f "${WORK_DIR}/${exp}/last_checkpoint" ]; then
        echo "skip: ${exp} already holds a checkpoint" >&2
        continue
    fi

    cmd=(sbatch -J "${job}" --gres="gpu:${GPU_TYPE}:${N_GPUS}"
         "${EXTRA_ARGS[@]}"
         "${SBATCH_SCRIPT}" "${ds}" "${n}" "${seed}" "${BACKBONE}")
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%s\n' "${cmd[*]}"
    else
        echo "submit: ${job} (drop ${n}, seed ${seed})"
        "${cmd[@]}"
        sleep 0.3
    fi
    count=$((count + 1))
done

echo "${count} job(s) ${DRY_RUN:+would be }submitted."
