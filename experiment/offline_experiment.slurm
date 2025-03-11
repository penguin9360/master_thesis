#!/bin/bash
#SBATCH --job-name=10k_inference_50k
#SBATCH --partition="gpu-medium"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=70GB 
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/offline_run/10k_inference_50k_%A.out
#SBATCH --gres=gpu:1

source activate aizynth-dev
module load CUDA/12.3.0
python 10k_inference_50k.py
