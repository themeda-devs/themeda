#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=persistence

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 20:00:00

#SBATCH -p physical
# SBATCH -p gpu-a100
# SBATCH --gres=gpu:1
#SBATCH --account=punim1932

# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load python/3.9.6
# module load cuda/11.6.0
module load web_proxy/latest

# Just for debugging
export CHIPLET_DIR=/data/gpfs/projects/punim1932/Data/chiplets2000/


# unzip to ssd
#export UNZIP_DISABLE_ZIPBOMB_DETECTION=true
#export CHIPLET_DIR=/tmp/chiplets
#for category in rain ; do 
#    export CATEGORY_DIR=$CHIPLET_DIR/$category
#    mkdir -p $CATEGORY_DIR
#    for year in $(seq 1988 2018) ; do 
#        for x in $( ls -1 /data/gpfs/projects/punim1932/Data/chiplets/${category}/ecofuture_chiplet_${category}_${year}_subset_* ) ; do 
#            unzip $x -d ${CATEGORY_DIR}
#        done
#    done
#done


poetry run python scripts/rain-persistence.py $CHIPLET_DIR/rain
