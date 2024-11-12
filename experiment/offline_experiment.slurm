#!/bin/bash
#SBATCH --job-name=10k_inf_200k
#SBATCH --partition="gpu-short"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=30GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/inference/10k_inf_200k_%A.out
#SBATCH --gres=gpu:1

source activate aizynth-dev
module load CUDA/12.3.0
python start_experiment.py