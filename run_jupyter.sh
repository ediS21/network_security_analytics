#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=jupyter
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # Request one GPU

# Change to the directory where your dataset is located
cd /home1/s4116488/jobs/network_security_analytics

# Activate the virtual environment
source /home1/s4116488/miniconda3/etc/profile.d/conda.sh
conda activate lstmenv

# Start Jupyter Notebook
jupyter notebook --no-browser --ip=$(hostname)