#!/bin/bash

#SBATCH --job-name=compute_grad_corr-2
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=/projects/PUCHALLA/LLP2024/tumor/logs/grad-corr-logs/%x.%j.out
#SBATCH --error=/projects/PUCHALLA/LLP2024/tumor/logs/grad-corr-logs/%x.%j.err

cd /projects/PUCHALLA/LLP2024/tumor

# conda init in ~/.bashrc is stale (points at a removed micromamba); activate the
# env directly by putting its bin on PATH so python resolves correctly.
export CONDA_PREFIX=/home/am0532/.conda/envs/sensaug
export PATH=${CONDA_PREFIX}/bin:$PATH

module load cudatoolkit/12.6 2>/dev/null || true

# Wrapper only -- no CLI flags here. All settings (WORK_DIR,
# RUN_KID_CALIBRATION, FIXED_MAGNITUDE_ONLY, etc.) live in
# scripts/run_grad_corr.sh itself; edit that file before submitting.
bash scripts/run_grad_corr.sh
