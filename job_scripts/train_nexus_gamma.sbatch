#!/bin/bash

#SBATCH --job-name=ours-pruned-loveda-1
#SBATCH --nodes=1

## GAMMA training config (per the lab's train_generic.sh guidelines)
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
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

# ---------------------------------------------------------------- config ----
# All overridable via env vars.
NAME="${NAME:-ours-pruned-loveda-1}"

# Valid --aug-type values (train.py argparse choices):
#   none ours default grad_corr random autoaugment augmix randaugment
#   trivialaugment idbh vip
# "sensaug" is NOT one of them -- train.py rejects it at argparse before any
# setup runs. The sensitivity-informed arm is `ours`; `grad_corr` additionally
# runs the gradient cross-correlation pipeline (add --no-corr-sa for its control
# arm: no SA, probes at the fixed 0.5 reference magnitude).
AUG_TYPE="${1:-ours}"
BACKBONE="${2:-segformer}"
DATASET="${3:-loveda}"

# Validate here rather than letting train.py's argparse catch it: under
# --launcher=pytorch that error prints once per rank (4x on this script) and
# only after torchrun has already spun up, which is a noisy and roundabout
# way to learn AUG_TYPE was misspelled.
VALID_AUG_TYPES=(none ours grad_corr default random autoaugment augmix randaugment trivialaugment idbh vip)
aug_type_ok=0
for valid in "${VALID_AUG_TYPES[@]}"; do
    if [ "${AUG_TYPE}" = "${valid}" ]; then
        aug_type_ok=1
        break
    fi
done
if [ "${aug_type_ok}" -ne 1 ]; then
    echo "ERROR: AUG_TYPE='${AUG_TYPE}' is not a valid --aug-type." >&2
    echo "Valid values: ${VALID_AUG_TYPES[*]}" >&2
    exit 1
fi

# Name includes AUG_TYPE so different aug types (or none vs grad_corr) never
# collide on the same work_dir -- train.py's resume check would otherwise
# silently continue a previous run's checkpoint under a different config.
# Keep "ours" in the name: train.py appends "_ours" to any exp_name lacking it when
# --aug-type=ours, which would desync EXP_NAME from the real work_dir (and the guard below).
# RUN_TAG suffixes the name for repeat runs of the same config, e.g. RUN_TAG=seed2.
# The `null` random-pruning control arm: drop RANDOM_PRUNE_COUNT ops chosen at
# random, once, and keep them dropped for the whole run. Unset by default, which
# is no random prune at all. To match a finished mRMR stage-1 run, pass its prune
# count:
#
#   RANDOM_PRUNE_COUNT=$(wc -w < experiments/<stage1>/mrmr_pruned_ops.txt) \
#     sbatch job_scripts/train_nexus_gamma.sbatch ours segformer ade20k
#
# The seed defaults to the SLURM job id -- unique per run, and identical across
# all four ranks, which is what matters: every rank runs train.py independently
# and a per-rank seed would train a different bank on each GPU. It is passed
# EXPLICITLY rather than left for train.py to derive, so EXP_NAME below can carry
# it (train.py appends the same suffix, and a mismatch would desync EXP_NAME from
# the real work_dir and defeat the resume guard). Set it by hand to repeat a draw
# or to sweep several draws of the same size: RANDOM_PRUNE_SEED=1, 2, ...
RANDOM_PRUNE_COUNT="${RANDOM_PRUNE_COUNT:-}"
RANDOM_PRUNE_SEED="${RANDOM_PRUNE_SEED:-${SLURM_JOB_ID}}"

EXP_NAME="${4:-${AUG_TYPE}_${NAME}_${BACKBONE}_${DATASET}_4gpu${RUN_TAG:+_${RUN_TAG}}}"
# Two random draws of the same size at different seeds are different
# experiments, so both the count and the seed go in the name -- otherwise they
# share a work_dir, interleave their logs, and the ALLOW_RESUME guard below
# (which keys off ${WORK_DIR}/${EXP_NAME}/last_checkpoint) has them silently
# resume each other. Matches the suffix train.py appends, so the two agree.
if [ -n "${RANDOM_PRUNE_COUNT}" ]; then
    if [ -z "${RANDOM_PRUNE_SEED}" ]; then
        # Only reachable outside SLURM, where SLURM_JOB_ID is unset. Refuse
        # rather than build an EXP_NAME ending in "_s" and let train.py pick its
        # own seed, which would put a different seed in the run than in its name.
        echo "ERROR: RANDOM_PRUNE_COUNT is set but RANDOM_PRUNE_SEED is empty" >&2
        echo "       (no SLURM_JOB_ID to derive it from). Set RANDOM_PRUNE_SEED." >&2
        exit 1
    fi
    EXP_NAME="${EXP_NAME}_nullprune${RANDOM_PRUNE_COUNT}_s${RANDOM_PRUNE_SEED}"
fi

WORK_DIR="${WORK_DIR:-./experiments}"
# gamma is a UMD Nexus partition/account -- use the Nexus cluster config, not della's.
CONFIG="${CONFIG:-configs/nexus.yaml}"

# Which function turns the redundancy score into the reweighted training pdf.
# train.py REQUIRES it on every --aug-type=grad_corr run (there is no default, so
# an arm is never left unnamed) and warns that it is ignored on every other arm --
# hence conditional rather than always appended.
DOWNWEIGHT="${DOWNWEIGHT:-mRMR}"
# Budget/tilt strength for Lever 3. train.py defaults this to 0.0, which is the
# short-circuit control arm (bit-identical to no downweighting, for either
# method) -- so it must be set explicitly here or every grad_corr run silently
# no-ops regardless of --corr-downweight-method. 0.25 is the calibrated value
# from CLAUDE.md / scripts/calibrate_lambda.py: ~2.3-3.1x tilt under
# soft-weighting, ~80% of ops survive the prune under mRMR.
CORR_LAMBDA="${CORR_LAMBDA:-0.25}"
EXTRA_ARGS=()
if [ "${AUG_TYPE}" = "grad_corr" ]; then
    EXTRA_ARGS+=(--corr-downweight-method="${DOWNWEIGHT}" --corr-lambda="${CORR_LAMBDA}")
fi

# Space-separated op names to permanently exclude from sampling for this
# entire run, on top of (actually: instead of -- it overrides, not merges)
# whatever configs/nexus.yaml's pruned_augmentations: already lists. Empty by
# default, meaning "use the config file's list as-is". train.py validates
# every name against the full augmentation vocabulary
# (sensaug/dataset/augmentations.py's LEGACY20_OPS/DIFF32_OPS) and refuses to
# start on an unrecognized one, so a typo here fails before any GPU is
# touched rather than mid-run.
#
# Available augmentation names (source of truth: sensaug/dataset/augmentations.py) --
#   LEGACY20_OPS (20, PascalCase -- used by --aug-type=random):
#     BrightnessTransform, NegativeBrightnessTransform, ColorTransform,
#     NegativeColorTransform, ContrastTransform, NegativeContrastTransform,
#     SharpnessTransform, NegativeSharpnessTransform, PosterizeTransform,
#     SolarizeTransform, ShearX, NegativeShearX, ShearY, NegativeShearY,
#     TranslateX, NegativeTranslateX, TranslateY, NegativeTranslateY,
#     Rotate, NegativeRotate
#   DIFF32_OPS / NON_DIFF32_OPS (32 shared snake_case names -- used by
#   --aug-type in ours/grad_corr/default; DIFF32_OPS is the GPU-batched
#   version, NON_DIFF32_OPS the CPU one, same keys):
#     lighter_R, darker_R, lighter_G, darker_G, lighter_B, darker_B,
#     lighter_H, darker_H, lighter_S, darker_S, lighter_V, darker_V,
#     blur, noise, rotate_pos, rotate_neg, shear_x_pos, shear_x_neg,
#     shear_y_pos, shear_y_neg, translate_x_pos, translate_x_neg,
#     translate_y_pos, translate_y_neg, brightness_pos, brightness_neg,
#     contrast_pos, contrast_neg, sharpness_pos, sharpness_neg,
#     color_pos, color_neg
# A name only exists in one of the two vocabularies above -- pruning a
# snake_case name while --aug-type=random (or vice versa) is accepted but
# logged as inert for that run (see train.py's warn_ignored_pruned_augmentations).
PRUNED_AUGS="${PRUNED_AUGS-darker_R darker_S lighter_G sharpness_neg sharpness_pos}"
if [ -n "${PRUNED_AUGS}" ]; then
    # Deliberately unquoted: it must word-split into separate argv entries
    # for train.py's `--pruned-augmentations` (nargs="+"), same shape as
    # TORCHRUN_ARGS above.
    EXTRA_ARGS+=(--pruned-augmentations ${PRUNED_AUGS})
fi

if [ -n "${RANDOM_PRUNE_COUNT}" ]; then
    EXTRA_ARGS+=(--random-prune-method=null
                 --random-prune-count="${RANDOM_PRUNE_COUNT}"
                 --random-prune-seed="${RANDOM_PRUNE_SEED}")
fi

# Your own checkout -- set REPO_DIR to override if it differs from this.
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"

# Group this job's bootstrap logs (see the #SBATCH --output/--error comment
# above) into job_scripts/logs/<job-name>/ before doing anything else, so a
# crash further down still leaves clean, grouped logs rather than flat ones.
# Rename, not copy: the fd SLURM already opened for stdout/stderr keeps
# writing to the same inode after the mv, so nothing written so far is lost
# and nothing later needs an explicit redirect. Anchored to the same absolute
# base as the #SBATCH paths above, so this works regardless of where `sbatch`
# was actually invoked from.
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
    echo "Pass a fresh exp name (4th arg) or RUN_TAG=..., or set ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

# `conda` itself is not on PATH on a fresh gamma compute-node shell -- this job
# previously only worked by inheriting miniforge's bin dir from the submitting
# shell's PATH via SLURM's `--export=ALL` default, which fails differently
# (and confusingly) the first time it's submitted from a shell that never
# activated conda itself. Load the module explicitly instead of relying on that.
# `module` is a shell function only an interactive login shell exports; a job
# submitted from a script inherits none, so re-source it and fall back to conda's
# own init rather than dying with a bare 127.
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/ummodules.sh; do
        [ -f "$f" ] && source "$f"
    done
fi
module load miniforge3 2>/dev/null || true
command -v conda >/dev/null 2>&1 || \
    source /opt/common/miniforge3/latest/etc/profile.d/conda.sh
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

# ---------------------------------------------------------------- train ----
# The correlation pipeline is an --aug-type now, not a flag: to enable it here,
# pass grad_corr as the first positional arg (or set AUG_TYPE), rather than
# uncommenting a flag.
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
    echo "ERROR: training exited ${TRAIN_STATUS}; skipping evaluation." >&2
    exit $TRAIN_STATUS
fi

if [ -n "${SKIP_EVAL}" ]; then
    echo "SKIP_EVAL set; training done, not evaluating."
    exit 0
fi

EXP_DIR="${WORK_DIR}/${EXP_NAME}"

# EVAL=all_augs: a full SA val round over every diff32 op, the run's own prune
# ignored, at the fixed shared levels so every arm is scored at the same
# magnitudes. Use it for arms test_robust.py can't compare (--aug-type=none has
# no perturbation_set, so test_robust would sweep legacy20 instead of diff32).
if [ "${EVAL:-test_robust}" = "all_augs" ]; then
    torchrun "${TORCHRUN_ARGS[@]}" \
        scripts/eval_all_augmentations.py \
        --work-dir "${EXP_DIR}" \
        --sa-curve sensaug/testing/shared_levels_diff32.json \
        --launcher pytorch
    EVAL_STATUS=$?
    echo "All done: ${EXP_DIR} (eval exit ${EVAL_STATUS})"
    exit $EVAL_STATUS
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

echo "All done: ${EXP_DIR}"
