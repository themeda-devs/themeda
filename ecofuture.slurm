#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=ecofuture

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 28:00:00

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
# module load cuda/11.6.0
module load web_proxy/latest

poetry run poetry run ecofuture train  --level4 ../Data/Level4_chips/ --width 128 --height 128 --wandb