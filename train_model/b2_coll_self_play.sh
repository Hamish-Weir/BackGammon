#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 2
#SBATCH -o data/b2_gameplay.out

# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124
module load libs/cuda/12.4.1

# --- Run one game ---
python -u p2_coll_self_play.py
