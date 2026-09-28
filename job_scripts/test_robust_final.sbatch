#!/bin/bash

#SBATCH --job-name=test-robust-final
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --ntasks=4
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.err
#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

# test_robust.py on a run's FINAL checkpoint instead of best*.pth.
#
# Same eval as job_scripts/test_robust.sbatch, but its results must not land in
# experiments_eval/<run>/: test_robust.py skips any (perturbation, level) that
# already has a results json there, so it would silently reuse the best-ckpt
# numbers. Instead the run is staged as experiments_final/<run>/ (a symlink to
# its config only) and test_robust.py derives experiments_final_eval/<run>/.
#
# Positional: <exp_dir> under experiments/. Env: CKPT (default iter_20000.pth,
# relative to exp_dir), REPO_DIR, SENSAUG_ENV. Resumable like test_robust.sbatch.
set -o pipefail

EXP_DIR="${1:?usage: sbatch test_robust_final.sbatch <exp_dir>}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"
CKPT="${CKPT:-iter_20000.pth}"

LOG_BASE="/gammascratch/amodak/SensAug/job_scripts/logs"
LOG_GROUP_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-test-robust-final}"
mkdir -p "${LOG_GROUP_DIR}"
for ext in out err; do
    BOOT="${LOG_BASE}/${SLURM_JOB_NAME:-test-robust-final}.${SLURM_JOB_ID}.${ext}"
    [ -f "${BOOT}" ] && mv "${BOOT}" "${LOG_GROUP_DIR}/${SLURM_JOB_NAME:-test-robust-final}.${SLURM_JOB_ID}.${ext}"
done

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
EXP_DIR="$(realpath "${EXP_DIR}")"
RUN="$(basename "${EXP_DIR}")"
CKPT_PATH="${EXP_DIR}/${CKPT}"
[ -f "${CKPT_PATH}" ] || { echo "ERROR: no checkpoint ${CKPT_PATH}" >&2; exit 1; }

CFGS=("${EXP_DIR}"/*.py)
[ "${#CFGS[@]}" -eq 1 ] && [ -f "${CFGS[0]}" ] || {
    echo "ERROR: expected exactly one config .py in ${EXP_DIR}" >&2; exit 1;
}
STAGE_DIR="${REPO_DIR}/experiments_final/${RUN}"
mkdir -p "${STAGE_DIR}"
ln -sfn "${CFGS[0]}" "${STAGE_DIR}/$(basename "${CFGS[0]}")"

SENSAUG_ENV="${SENSAUG_ENV:-/nfshomes/amodak/.conda/envs/sensaug}"
[ -x "${SENSAUG_ENV}/bin/python" ] || { echo "ERROR: no python at ${SENSAUG_ENV}/bin/python" >&2; exit 1; }
export PATH="${SENSAUG_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SENSAUG_ENV}"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh 2>/dev/null || true
command -v module >/dev/null 2>&1 && module load cuda/11.7.0 2>/dev/null || true

echo "python:   $(command -v python)  ($(python --version 2>&1))"
echo "exp_dir:  ${EXP_DIR}"
echo "ckpt:     ${CKPT_PATH}"
echo "results:  ${REPO_DIR}/experiments_final_eval/${RUN}"
export OMP_NUM_THREADS=16

python test_robust.py --work-dir "${STAGE_DIR}" --checkpoint "${CKPT_PATH}"
