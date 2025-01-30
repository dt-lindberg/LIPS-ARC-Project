#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00 
#SBATCH --job-name=InstallEnvironment
#SBATCH --output=slurm_output_conda_env_install.out

# Loading modules
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/lost_in_program_space/MoL-ARC-AGI

# Install both conda environments
conda env create -f environment.yml
conda env create -f environment_2.yml
