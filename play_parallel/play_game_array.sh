#!/bin/bash --login

#SBATCH -p serial
#SBATCH --array=1-1000
#SBATCH -t 20
#SBATCH -o logs/game_%A_%a.out

# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124
module load libs/cuda/12.4.1

# --- Run one game ---
echo "Running game with $v1 vs $v2"

python -u play_single_game.py "$v1" "$v2"
