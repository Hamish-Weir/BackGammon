#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 10
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix

# --- Create BoardStates ---
python -u p1_1_generate_boardstates.py