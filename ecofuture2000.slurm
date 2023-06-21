#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=ecofuture

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 04:00:00

# SBATCH -p physical
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=punim1932

# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables

module purge
module load python/3.9.6
module load cuda/11.7.0
module load cudnn/8.8.1.3-cuda-11.7.0
module load nccl/2.14.3-cuda-11.7.0
module load web_proxy/latest
export CHIPLET_DIR=/data/gpfs/projects/punim1932/Data/chiplets2000/
export BATCH=4
export SMOOTHING=.1
export MAX_YEARS=2
# export RUN_NAME=eco-b$BATCH-sm$SMOOTHING-y$MAX_YEARS-lr-5
export RUN_NAME=test-multi
poetry run ecofuture train --level4 $CHIPLET_DIR/level4 --rain $CHIPLET_DIR/rain --tmax $CHIPLET_DIR/tmax --validation-subset 1 --max-chiplets 20 --batch-size $BATCH  --run-name $RUN_NAME --output-dir outputs/$RUN_NAME --max-years $MAX_YEARS --learning-rate 0.00001 --temporal-processor-type GRU