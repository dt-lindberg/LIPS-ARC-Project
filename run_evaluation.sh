#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --job-name=LIPSeval
#SBATCH --output=slurm_output_%A.out

# Loading modules (CUDA and Anaconda are located in module 2024)
module load 2024
module load CUDA/12.6.0
module load Anaconda3/2024.06-1

# Activate conda environment for the induction model
source activate lips_env2

# Set working directory for induction
cd $HOME/lost_in_program_space/MoL-ARC-AGI/induction-agi

# Logging info
echo "Starting job at $(date)"

# Run the induction model
srun python Induction_part/run_induction.py
# >> will create a induction_results.json file in a results folder

echo "Induction model finished at $(date)"

# Change working directory
cd "$TMPDIR" # $TMPDIR is set to /scratch-local/username

# Copy transduction data to local scratch
if cp -r $HOME/lost_in_program_space/MoL-ARC-AGI/data "$TMPDIR"; then
    echo "data copied to scratch: $HOME/lost_in_program_space/MoL-ARC-AGI/data --> $TMPDIR/data"
else
    echo "Error: Failed to copy data from $HOME/lost_in_program_space/MoL-ARC-AGI/data to $TMPDIR"
    exit 1
fi

# Copy induction results to local scratch
if cp -r $HOME/lost_in_program_space/MoL-ARC-AGI/induction-agi/results "$TMPDIR"; then
    echo "data copied to scratch: $HOME/lost_in_program_space/MoL-ARC-AGI/induction-agi/results --> $TMPDIR/results"
else
    echo "Error: Failed to copy data from $HOME/lost_in_program_space/MoL-ARC-AGI/induction-agi/results to $TMPDIR"
    exit 1
fi


# Reload the second conda environment for the transduction model
conda deactivate
conda activate lips_env

# Run the python script
srun python $HOME/lost_in_program_space/MoL-ARC-AGI/training_code/run_evaluation_Llama-rearc_with_ttt.py

# Copy output directory from scratch back to desired directory
cp -r output_evaluation_Llama-rearc_with_ttt $HOME/lost_in_program_space/MoL-ARC-AGI