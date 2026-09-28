#!/bin/bash

#SBATCH --job-name=eval-all-augs
#SBATCH --nodes=1

## GAMMA config
#SBATCH --time=03:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32gb
#SBATCH --ntasks=4

## For a faster full sweep on a big val set, request 4 GPUs and submit with
## LAUNCHER=pytorch:
##SBATCH --gres=gpu:rtxa6000:4
##SBATCH --ntasks=16

## Tron fallback
##SBATCH --time=02:59:00
##SBATCH --qos=high
##SBATCH --account=nexus
##SBATCH --partition=tron
##SBATCH --gres=gpu:rtxa4000:1
##SBATCH --mem=32gb
##SBATCH --ntasks=4

# SLURM opens these before the body runs; keep them absolute and flat, then
# group into a per-job-name subdir once running (same-fs rename keeps the open
# fd valid). Mirrors job_scripts/train_nexus_gamma.sbatch.
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ---------------------------------------------------------------- config ----
# Positional: <exp_dir>. Everything else via env:
#   LAUNCHER=none|pytorch   (default none = 1 GPU plain python; SA search is
#                            single-process, so `none` is the intended mode)
#   CHECKPOINT_FLAG         (default --use-latest; set to "" for best*.pth, or
#                            e.g. "--checkpoint experiments/foo/iter_20000.pth")
#   EXTRA_ARGS             (e.g. "--ratio 0.1 --num-levels 3 --ops noise blur")
#   REPO_DIR, SENSAUG_ENV
# The script recovers the perturbation_set / ratio from the checkpoint's own
# embedded config -- nothing to pass.
EXP_DIR="${1:?usage: sbatch eval_all_augmentations.sbatch <exp_dir>}"
LAUNCHER="${LAUNCHER:-none}"
CHECKPOINT_FLAG="${CHECKPOINT_FLAG---use-latest}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"

LOG_BASE="/gammascratch/amodak/SensAug/job_scripts/logs"
LOG_GROUP_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-eval-all-augs}"
mkdir -p "${LOG_GROUP_DIR}"
for ext in out err; do
    BOOT="${LOG_BASE}/${SLURM_JOB_NAME:-eval-all-augs}.${SLURM_JOB_ID}.${ext}"
    [ -f "${BOOT}" ] && mv "${BOOT}" "${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-eval-all-augs}.${SLURM_JOB_ID}.${ext}"
done

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
[ -f scripts/eval_all_augmentations.py ] || {
    echo "ERROR: no scripts/eval_all_augmentations.py under ${REPO_DIR}" >&2; exit 1;
}

# A batch shell here has neither `module` nor `conda` on PATH (profile.d is not
# sourced for non-login shells, and ~/.bashrc's conda init is stale). Put the
# env's bin dir on PATH directly -- same approach as job_scripts/compute_grad_corr.sbatch.
# Override with SENSAUG_ENV=/path/to/conda/env if your checkout differs.
SENSAUG_ENV="${SENSAUG_ENV:-/nfshomes/amodak/.conda/envs/sensaug}"
if [ ! -x "${SENSAUG_ENV}/bin/python" ]; then
    echo "ERROR: no python at ${SENSAUG_ENV}/bin/python; set SENSAUG_ENV" >&2
    exit 1
fi
export PATH="${SENSAUG_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SENSAUG_ENV}"
# Best-effort modules init, only if the site provides it -- torch 2.0.0+cu117
# bundles its own CUDA runtime, so a cuda module is not required.
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh 2>/dev/null || true
command -v module >/dev/null 2>&1 && module load cuda/11.7.0 2>/dev/null || true

echo "python: $(command -v python)  ($(python --version 2>&1))"
export OMP_NUM_THREADS=16

# The script recovers its config from the checkpoint meta -- no --cluster-config.
COMMON_ARGS=(--work-dir "${EXP_DIR}" ${CHECKPOINT_FLAG})

if [ "${LAUNCHER}" = "pytorch" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
    PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
    echo "torchrun on ${NUM_GPUS} GPU(s)"
    torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
        scripts/eval_all_augmentations.py "${COMMON_ARGS[@]}" \
        --launcher pytorch ${EXTRA_ARGS:-}
else
    echo "plain python, 1 GPU"
    python scripts/eval_all_augmentations.py "${COMMON_ARGS[@]}" ${EXTRA_ARGS:-}
fi
