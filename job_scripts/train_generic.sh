#!/bin/bash

# Define a base name for the job, experiment, and output files

#SBATCH --job-name=train_generic                            
#SBATCH --output=slurm_logs/%x.out.%j       # indicates a file to redirect STDOUT to; %j is the jobid
#SBATCH --error=slurm_logs/%x.out.%j        # indicates a file to redirect STDERR to; %j is the jobid

#SBATCH --mem=120gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16

# set up notification settings for failures
##SBATCH --mail-user=YOUREMAILHERE@umd.edu
##SBATCH --mail-type=ALL

## Scavenger training config
##SBATCH --time=48:00:00
##SBATCH --qos=scavenger
##SBATCH --account=scavenger
##SBATCH --partition=scavenger

## GAMMA training config
#SBATCH --time=48:00:00     
#SBATCH --qos=huge-long                                    
#SBATCH --account=gamma
#SBATCH --partition=gamma

################################
## Tron config
##SBATCH --time=23:59:00  
##SBATCH --qos=high
##SBATCH --account=nexus
##SBATCH --partition=tron

##SBATCH --mem=64gb                                               # memory required by job; if unit is not specified MB will be assumed
##SBATCH --gres=gpu:rtxa4000:4
##SBATCH --ntasks=15


eval "$(conda shell.bash hook)"
conda activate bp-38

module load cuda/11.7.0

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Number of GPUS: $NUM_GPUS"

# Exp name
WORK_ROOT_DIR="./experiments" # master folder for experiments
WORK_DIR_NAME="baselines" # subfolder
WORK_DIR="${WORK_ROOT_DIR}/${WORK_DIR_NAME}"

AUG_TYPE=$1
BACKBONE=$2
DATASET=$3

EXP_NAME=$4

# check if command line argument is empty or not present
if [ -z $4 ]; then
    EXP_NAME="${AUG_TYPE}_${BACKBONE}_${DATASET}"
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=16 \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --backbone=${BACKBONE} \
    --dataset=${DATASET} \
    --aug-type=${AUG_TYPE} \
    --work_dir=${WORK_DIR} \
    --exp_name=${EXP_NAME} \
    --launcher="pytorch" \
    --no-inv-aug \
    

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=16 \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    test_robust.py \
    --work-dir="${WORK_DIR}/${EXP_NAME}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best
    
# real world testing

if [[ "$DATASET" == "cityscapes" ]]; then
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    test.py \
    --work-dir="${WORK_DIR}/${EXP_NAME}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best \
    --test_data="acdc" \

    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    test.py \
    --work-dir="${WORK_DIR}/${EXP_NAME}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best \
    --test_data="idd" \

    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    test.py \
    --work-dir="${WORK_DIR}/${EXP_NAME}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best \
    --test_data="zurich" \

    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    test.py \
    --work-dir="${WORK_DIR}/${EXP_NAME}" \
    --launcher="pytorch" \
    --save-vis \
    --use-best \
    --test_data="night" \
    
fi
