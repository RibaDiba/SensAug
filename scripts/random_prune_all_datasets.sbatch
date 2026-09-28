#!/bin/bash

#SBATCH --job-name=ours-pruned-random
#SBATCH --nodes=1

## Same allocation as the ours-pruned runs this is the control for: every one of
## them (ours_ours-pruned-* and seeded_*_pruned_ours) ran on rtxa6000 x4 on gamma.
## random_prune_all_datasets.sh re-passes --gres explicitly, so this line is only
## the default for a hand submission.
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --mem=120gb
#SBATCH --ntasks=16

## Absolute, and flat: SLURM opens these before the body runs, so a per-job
## subdirectory can't exist yet. The body moves them into logs/<job-name>/.
#SBATCH --output=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/SensAug/job_scripts/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# random_prune_all_datasets.sbatch -- ONE dataset's random-pruning control.
#
# "ours pruned, but the pruned ops are chosen at random": --aug-type=ours with
# --random-prune-method=null dropping N ops, where the mRMR arm drops the N ops it
# ranked most redundant. It answers the question mRMR invites -- does ranking by
# redundancy beat dropping the same number at random? -- so it must differ from
# the pruned runs in WHICH ops are dropped and nothing else. Hence:
#
#   * The train.py flags below are the pruned runs' flags, verbatim
#     (--launcher=pytorch --no-inv-aug --auto-scale-lr --descending-MA), checked
#     against their own logs: hold_none_prob=False, descending_MA=True.
#   * NO --hold-none-prob. The pruned runs did not pass it, and CLAUDE.md is
#     explicit that it goes on both arms or neither.
#   * NO --pruned-augmentations, and deliberately NOT launched through
#     job_scripts/train_nexus_gamma.sbatch: its PRUNED_AUGS defaults to five ops
#     and, because ${VAR:-default} substitutes on empty as well as unset, cannot
#     be emptied from the environment. Going through it would drop those five
#     PLUS N random ones -- a control that prunes twice as much as its partner.
#   * The conda setup is grad_corr_seed.sbatch's, not train_nexus_gamma.sbatch's.
#     `module` is a shell function that a job only inherits when submitted from
#     an interactive login shell; this is submitted from a script, which is
#     exactly the case where a bare `module load` dies with exit 127 one second in.
#
# Usage (normally via random_prune_all_datasets.sh):
#   sbatch scripts/random_prune_all_datasets.sbatch <dataset> <count> <seed> [backbone]
#
# <seed> is required and never derived per job: all four ranks must draw the same
# ops, and the driver has to know the work_dir name in advance for its guards.
#
# Env knobs: WORK_DIR, CONFIG, CONDA_ENV, REPO_DIR, ALLOW_RESUME, SKIP_EVAL,
#   SAVE_VIS=1 (off by default -- see the eval section).
# ==========================================================================

DATASET="${1:?usage: <dataset> <count> <seed> [backbone]}"
COUNT="${2:?random-prune count required (arg 2)}"
SEED="${3:?random-prune seed required (arg 3)}"
BACKBONE="${4:-segformer}"

WORK_DIR="${WORK_DIR:-./experiments}"
CONFIG="${CONFIG:-configs/nexus.yaml}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/SensAug}"

# Pre-suffixed with exactly what train.py appends to every null-arm exp name
# (train.py: "nullprune{count}_s{seed}"), so its rename is a no-op and the resume
# guard below checks the directory the run will actually write to.
EXP_NAME="ours-pruned-random-${DATASET}_nullprune${COUNT}_s${SEED}"

LOG_BASE="/gammascratch/amodak/SensAug/job_scripts/logs"
LOG_GROUP_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-ours-pruned-random-${DATASET}}"
mkdir -p "${LOG_GROUP_DIR}"
for ext in out err; do
    BOOT="${LOG_BASE}/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.${ext}"
    [ -f "${BOOT}" ] && mv "${BOOT}" "${LOG_GROUP_DIR}/"
done

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
[ -f train.py ] || { echo "ERROR: no train.py under ${REPO_DIR}" >&2; exit 1; }

if [ -z "${ALLOW_RESUME}" ] && [ -f "${WORK_DIR}/${EXP_NAME}/last_checkpoint" ]; then
    echo "ERROR: ${WORK_DIR}/${EXP_NAME} already holds a checkpoint; train.py would resume it." >&2
    echo "       Use a different seed, or ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

# ------------------------------------------------------- environment setup ----
# Copied from job_scripts/grad_corr_seed.sbatch -- see the header for why.
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/modules.sh /etc/profile.d/ummodules.sh; do
        # shellcheck disable=SC1090
        [ -f "${_init}" ] && source "${_init}"
    done
fi
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
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found; set CONDA_SH=/path/to/etc/profile.d/conda.sh" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-sensaug}" || {
    echo "ERROR: could not activate conda env '${CONDA_ENV:-sensaug}'" >&2; exit 1; }
module load cuda/11.7.0 2>/dev/null \
    || echo "WARNING: could not 'module load cuda/11.7.0'; relying on torch's bundled runtime." >&2
command -v torchrun >/dev/null 2>&1 || {
    echo "ERROR: torchrun not on PATH after activating the env. PATH=${PATH}" >&2; exit 1; }

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}
# Derived from the job id so two jobs on one node don't both take 29500.
PORT=${PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}
TORCHRUN_ARGS=(
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    --nproc_per_node=$NUM_GPUS
    --master_port=$PORT
)
export OMP_NUM_THREADS=16

echo "=== ours-pruned-random: dataset=${DATASET} backbone=${BACKBONE}" \
     "drop=${COUNT} seed=${SEED} gpus=${NUM_GPUS} exp=${EXP_NAME} ==="

# ---------------------------------------------------------------- train ----
torchrun "${TORCHRUN_ARGS[@]}" \
    train.py \
    --cluster-config=${CONFIG} \
    --backbone=${BACKBONE} \
    --dataset=${DATASET} \
    --aug-type=ours \
    --work_dir=${WORK_DIR} \
    --exp_name=${EXP_NAME} \
    --launcher="pytorch" \
    --no-inv-aug \
    --auto-scale-lr \
    --descending-MA \
    --random-prune-method=null \
    --random-prune-count="${COUNT}" \
    --random-prune-seed="${SEED}"

TRAIN_STATUS=$?
if [ $TRAIN_STATUS -ne 0 ]; then
    echo "ERROR: training exited ${TRAIN_STATUS}; skipping evaluation." >&2
    exit $TRAIN_STATUS
fi
[ -n "${SKIP_EVAL}" ] && { echo "SKIP_EVAL set; not evaluating."; exit 0; }

EXP_DIR="${WORK_DIR}/${EXP_NAME}"

# ------------------------------------------------- robustness evaluation ----
# Same evals as the pruned runs, so the numbers are comparable. The one
# difference is --save-vis, OFF by default: it writes a PNG overlay per sample
# per perturbation, which is what filled /gammascratch to 0 bytes. It changes no
# metric, and a checkpoint can be re-evaluated with it at any time.
VIS_ARGS=()
[ -n "${SAVE_VIS}" ] && VIS_ARGS=(--save-vis)

torchrun "${TORCHRUN_ARGS[@]}" \
    test_robust.py \
    --work-dir="${EXP_DIR}" \
    --launcher="pytorch" \
    --use-best \
    "${VIS_ARGS[@]}"

# Cityscapes-trained models only: these all assume the 19-class label space.
if [[ "$DATASET" == "cityscapes" ]]; then
    for TEST_DATA in acdc idd zurich night; do
        echo "=== test.py on ${TEST_DATA} ==="
        torchrun "${TORCHRUN_ARGS[@]}" \
            test.py \
            --cluster-config=${CONFIG} \
            --work-dir="${EXP_DIR}" \
            --launcher="pytorch" \
            --use-best \
            --test_data="${TEST_DATA}" \
            "${VIS_ARGS[@]}" \
        || echo "WARNING: test.py --test_data=${TEST_DATA} failed; continuing." >&2
    done
fi

echo "All done: ${EXP_DIR}"
