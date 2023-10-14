#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=ecofuture

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 34:00:00

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
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.7.0

export PATH=/home/rturnbull/runting/poetry-py3.9.6/bin:$PATH

export BASE_DIR=/data/gpfs/projects/punim1932/data_wip


BATCH=1
LEARNING_RATE=0.001
KERNEL=15
HIDDEN=64
EMBEDDING=16


RUN_NAME=themeda-all

poetry run ecofuture train \
    --input land_cover --input rain --input tmax --input elevation --input land_use \
    --input fire_scar_early  --input fire_scar_late \
    --input soil_ece --input soil_clay --input soil_depth \
    --output land_cover \
    --validation-subset 1 --batch-size $BATCH \
    --learning-rate $LEARNING_RATE --temporal-processor-type GRU \
    --kernel-size $KERNEL --embedding-size $EMBEDDING \
    --base-dir $BASE_DIR \
    --run-name $RUN_NAME --output-dir outputs/$RUN_NAME \
    --max-years 25 \
    --wandb --wandb-entity punim1932
