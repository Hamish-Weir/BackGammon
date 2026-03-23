#!/bin/bash --login

#SBATCH -p serial
#SBATCH --array=1-1000
#SBATCH -t 60
#SBATCH -o data/eval_play_games_M_R/game_%A_%a.out

# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124
module load libs/cuda/12.4.1

# --- Run one game ---
python -u p4_play_eval_play.py "A0" "MCTS" "data/eval_play_games_M_R/$SLURM_ARRAY_TASK_ID.txt"
