#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
# SBATCH --job-name=tmax-persistence

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 20:00:00

#SBATCH -p physical
# SBATCH -p gpu-a100
# SBATCH --gres=gpu:1
#SBATCH --account=punim1932

# Specify your email address to be notified of progress.
SBATCH --mail-user=akshay.gohil@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load python/3.9.6
module load cuda/11.7.0
module load cudnn/8.8.1.3-cuda-11.7.0
module load nccl/2.14.3-cuda-11.7.0
module load web_proxy/latest

# Just for debugging
# export CHIPLET_DIR=/data/gpfs/projects/punim1932/Data/chiplets2000/


# unzip to ssd
export UNZIP_DISABLE_ZIPBOMB_DETECTION=true
export CHIPLET_DIR=/tmp/chiplets
for category in tmax ; do 
   export CATEGORY_DIR=$CHIPLET_DIR/$category
   mkdir -p $CATEGORY_DIR
   for year in $(seq 1988 2018) ; do 
       for x in $( ls -1 /data/gpfs/projects/punim1932/Data/chiplets/${category}/themeda_chiplet_${category}_${year}_subset_* ) ; do 
           unzip $x -d ${CATEGORY_DIR}
       done
   done
done


poetry run python scripts/tmax-persistence.py $CHIPLET_DIR/tmax
