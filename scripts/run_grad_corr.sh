#!/bin/bash
# Reference invocation for scripts/calibrate_kid_magnitudes.py and
# scripts/compute_grad_corr.py against an already-trained experiment.
# Edit the variables below, then run: bash scripts/run_grad_corr.sh

set -e

# Run from the repo root regardless of where this script is invoked from, so
# WORK_DIR and the python scripts/*.py paths below resolve correctly.
cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

eval "$(conda shell.bash hook)"
conda activate sensaug

# ---------------------------------------------------------------------------
# experiments/<exp_name> -- must already contain a dumped config (*.py) and
# at least one checkpoint (best*.pth or last_checkpoint).
WORK_DIR="experiments/grad_corr_grad_corr_test_3_pspnet_cityscapes_1gpu_gradcorr"

# Set to 1 to run scripts/calibrate_kid_magnitudes.py first and feed its
# output into compute_grad_corr.py via --magnitudes-path. Set to 0 to skip
# calibration and let compute_grad_corr.py use its own default (a live SA
# run's corr_magnitudes.json if the experiment has one, else the fixed
# ref-magnitude fallback).
RUN_KID_CALIBRATION=0

# Set to 1 to skip KID calibration entirely (no extra memory/compute for the
# grid search) and force every op to the flat COMPUTE_REF_MAGNITUDE (0.5) --
# overrides RUN_KID_CALIBRATION and ignores any corr_magnitudes.json the
# experiment already has, so R still comes out, just without a commensurate
# per-op magnitude.
FIXED_MAGNITUDE_ONLY=1

# =============================================================================
# scripts/calibrate_kid_magnitudes.py flags
# =============================================================================

# Clean reference image directory KID is computed against. Default (leave
# empty) is the val set from WORK_DIR's dumped config.
KID_REFERENCE_DIR=""

# Shared target KID, derived by running this op at this magnitude once
# ("<op>@<magnitude>"). Overridden by KID_TARGET if that is set.
KID_TARGET_FROM="noise@0.5"

# Explicit shared target KID value (float). Leave empty to use KID_TARGET_FROM
# instead.
KID_TARGET=""

# Number of candidate magnitudes in [0.05, 1.0] evaluated per op.
KID_GRID_SIZE=8

# Subset of op names to calibrate (space-separated), e.g. "noise blur".
# Leave empty for all differentiable ops.
KID_OPS=""

# Images per KID subset.
KID_BATCH_SIZE=25

# Where calibrate_kid_magnitudes.py writes its seed file. Default (leave
# empty) is "$WORK_DIR/corr_magnitudes_kid_seed.json".
KID_OUTPUT=""

# "cuda" or "cpu". Default (leave empty) picks cuda if available.
KID_DEVICE=""

# =============================================================================
# scripts/compute_grad_corr.py flags
# =============================================================================

# Explicit checkpoint path. Leave empty to auto-pick best*.pth in WORK_DIR
# (or last_checkpoint if COMPUTE_USE_LATEST=1).
COMPUTE_CHECKPOINT=""

# Use the checkpoint pointed to by last_checkpoint instead of best*.pth.
COMPUTE_USE_LATEST=0

# Where aug_gradient_log.txt / corr_matrix_log.json / corr_bootstrap_log.txt
# get written. Default (leave empty) is WORK_DIR itself (in place).
COMPUTE_OUTPUT_DIR=""

# Images per forward during the sweep.
COMPUTE_SWEEP_BATCH_SIZE=1

# mode | sampled_shared | sampled_independent | fixed
COMPUTE_MAGNITUDE_MODE="mode"

# corr_magnitudes.json-shaped seed file. If RUN_KID_CALIBRATION=1 this gets
# overridden below to point at the KID script's output; otherwise, leave
# empty to auto-detect "$WORK_DIR/corr_magnitudes.json" (a live SA run's own
# log) if it exists, else fall back to COMPUTE_REF_MAGNITUDE for every op.
COMPUTE_MAGNITUDES_PATH=""

# Fallback magnitude for any op the magnitudes-path snapshot doesn't cover
# (and for every op when there is no snapshot at all).
COMPUTE_REF_MAGNITUDE=0.5

# RNG seed applied before each probe batch.
COMPUTE_PROBE_SEED=0

# Minimum images in the probe window before R is computed at all.
COMPUTE_N_MIN=100

# Skip cluster-bootstrap CIs + BH-FDR (faster). 1 to disable, 0 to keep on.
COMPUTE_NO_BOOTSTRAP=0

# Bootstrap replicates, if bootstrap is enabled.
COMPUTE_BOOTSTRAP_REPS=1000

# =============================================================================

if [ "$FIXED_MAGNITUDE_ONLY" = "1" ]; then
    RUN_KID_CALIBRATION=0
    COMPUTE_MAGNITUDE_MODE="fixed"
    COMPUTE_MAGNITUDES_PATH=""
fi

if [ "$RUN_KID_CALIBRATION" = "1" ]; then
    kid_args=(--work-dir "$WORK_DIR")
    [ -n "$KID_REFERENCE_DIR" ] && kid_args+=(--kid-reference-dir "$KID_REFERENCE_DIR")
    [ -n "$KID_TARGET" ] && kid_args+=(--target-kid "$KID_TARGET") || kid_args+=(--target-kid-from "$KID_TARGET_FROM")
    kid_args+=(--grid-size "$KID_GRID_SIZE")
    [ -n "$KID_OPS" ] && kid_args+=(--ops $KID_OPS)
    kid_args+=(--batch-size "$KID_BATCH_SIZE")
    KID_OUTPUT="${KID_OUTPUT:-$WORK_DIR/corr_magnitudes_kid_seed.json}"
    kid_args+=(--output "$KID_OUTPUT")
    [ -n "$KID_DEVICE" ] && kid_args+=(--device "$KID_DEVICE")

    python scripts/calibrate_kid_magnitudes.py "${kid_args[@]}"

    COMPUTE_MAGNITUDES_PATH="$KID_OUTPUT"
fi

compute_args=(--work-dir "$WORK_DIR")
[ -n "$COMPUTE_CHECKPOINT" ] && compute_args+=(--checkpoint "$COMPUTE_CHECKPOINT")
[ "$COMPUTE_USE_LATEST" = "1" ] && compute_args+=(--use-latest)
[ -n "$COMPUTE_OUTPUT_DIR" ] && compute_args+=(--output-dir "$COMPUTE_OUTPUT_DIR")
compute_args+=(--sweep-batch-size "$COMPUTE_SWEEP_BATCH_SIZE")
compute_args+=(--magnitude-mode "$COMPUTE_MAGNITUDE_MODE")
[ -n "$COMPUTE_MAGNITUDES_PATH" ] && compute_args+=(--magnitudes-path "$COMPUTE_MAGNITUDES_PATH")
compute_args+=(--ref-magnitude "$COMPUTE_REF_MAGNITUDE")
compute_args+=(--probe-seed "$COMPUTE_PROBE_SEED")
compute_args+=(--n-min "$COMPUTE_N_MIN")
[ "$COMPUTE_NO_BOOTSTRAP" = "1" ] && compute_args+=(--no-bootstrap)
compute_args+=(--bootstrap-reps "$COMPUTE_BOOTSTRAP_REPS")

python scripts/compute_grad_corr.py "${compute_args[@]}"
