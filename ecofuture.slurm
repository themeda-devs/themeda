#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=ecofuture

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 20:00:00

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
export UNZIP_DISABLE_ZIPBOMB_DETECTION=true
export CHIPLET_DIR=/tmp/chiplets
for category in level4 rain tmax ; do 
    export CATEGORY_DIR=$CHIPLET_DIR/$category
    mkdir -p $CATEGORY_DIR
    for year in $(seq 1988 2018) ; do 
        for x in $( ls -1 /data/gpfs/projects/punim1932/Data/chiplets/${category}/ecofuture_chiplet_${category}_${year}_subset_* ) ; do 
            unzip $x -d ${CATEGORY_DIR}
        done
    done
done

# export CHIPLET_DIR=/data/gpfs/projects/punim1932/Data/chiplets2000/

export BATCH=4
export SMOOTHING=0
export MAX_YEARS=10
export RUN_NAME=eco-b$BATCH-sm$SMOOTHING-y$MAX_YEARS-lr-4-B

poetry run ecofuture train --level4 $CHIPLET_DIR/level4 --rain $CHIPLET_DIR/rain --tmax $CHIPLET_DIR/tmax --validation-subset 1 --batch-size $BATCH  --run-name $RUN_NAME --output-dir outputs/$RUN_NAME --max-years $MAX_YEARS --learning-rate 0.0001 --temporal-processor-type GRU --wandb # --wandb-entity punim1932

