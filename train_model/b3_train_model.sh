#!/bin/bash --login

#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 1
#SBATCH -o data/b3_training.out


# --- Load Modules ---
module purge
module load apps/binapps/conda/miniforge3/25.3.0_python3.10
module load apps/binapps/anaconda3/2019.07-numpy-fix
module load apps/binapps/pytorch/2.6.0-312-gpu-cu124
module load libs/cuda/12.4.1

#---Train Model---
python -u p3_train_model.py
