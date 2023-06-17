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

# unzip to ssd
export CHIPLET_DIR=/tmp/chiplets
export UNZIP_DISABLE_ZIPBOMB_DETECTION=true
for x in $(find /data/gpfs/projects/punim1932/Data/chiplets/level4/ -mindepth 1 -maxdepth 1 -type f -perm -444) ; do
    unzip $x -d $CHIPLET_DIR
done

# export CHIPLET_DIR=/data/gpfs/projects/punim1932/Data/chiplets2000/



export BATCH=4
export SMOOTHING=.1
export MAX_YEARS=15
# export RUN_NAME=eco-b$BATCH-sm$SMOOTHING-y$MAX_YEARS-lr-5
export RUN_NAME=test-multi

poetry run ecofuture train --level4 $CHIPLET_DIR/level4 --rain $CHIPLET_DIR/rain --tmax $CHIPLET_DIR/tmax --validation-subset 1 --max-chiplets 10000 --batch-size $BATCH  --run-name $RUN_NAME --output-dir outputs/$RUN_NAME --max-years 6 --learning-rate 0.00001 --temporal-processor-type GRU --wandb # --wandb-entity punim1932

#poetry run poetry run ecofuture train --chiplet-dir $CHIPLET_DIR --validation-subset 1 --max-chiplets 0 --label-smoothing $SMOOTHING --batch-size $BATCH  --run-name $RUN_NAME --output-dir outputs/$RUN_NAME --max-years $MAX_YEARS --learning-rate 0.00001 --wandb # --wandb-entity punim1932

# poetry run poetry run ecofuture train --chiplet-dir $CHIPLET_DIR --validation-subset 1 --max-chiplets 0 --label-smoothing $SMOOTHING --batch-size $BATCH  --run-name $RUN_NAME --output-dir outputs/$RUN_NAME --max-years $MAX_YEARS --learning-rate 0.00001 --wandb # --wandb-entity punim1932