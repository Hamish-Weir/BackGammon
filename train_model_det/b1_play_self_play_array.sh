#!/bin/bash --login

#SBATCH -p serial
#SBATCH --array=1-400
#SBATCH -t 60
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124
module load libs/cuda/12.4.1

# --- Run one game ---
# epoch=1
python -u p1_play_self_play.py $((epoch % 3)) $SLURM_ARRAY_TASK_ID
