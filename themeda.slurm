#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=themeda

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 42:00:00

#SBATCH --mem-per-cpu=8G

# SBATCH -p physical
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=punim1932

# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load foss/2022a
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load NCCL/2.18.3-CUDA-12.2.0

export PATH=/home/rturnbull/runting/poetry-py3.9.6/bin:$PATH
export THEMEDA_DATA_DIR=/data/gpfs/projects/punim1932/data

BATCH=8
LEARNING_RATE=0.001
KERNEL=15
HIDDEN=64
EMBEDDING=16

TEMPORAL=LSTM

RUN_NAME=themeda-outLC-nov9

poetry run themeda train \
    --input land_cover --input rain --input tmax --input elevation --input land_use \
    --input soil_ece --input soil_clay --input soil_depth \
    --input fire_scar_early  --input fire_scar_late \
    --output land_cover \
    --validation-subset 1 --batch-size $BATCH \
    --learning-rate $LEARNING_RATE --temporal-processor-type $TEMPORAL \
    --kernel-size $KERNEL --embedding-size $EMBEDDING \
    --base-dir $THEMEDA_DATA_DIR \
    --run-name $RUN_NAME --output-dir outputs/$RUN_NAME \
    --no-emd-loss --no-hierarchical-embedding \
    --wandb --wandb-entity punim1932


