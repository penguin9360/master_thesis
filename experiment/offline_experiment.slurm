#!/bin/bash
#SBATCH --job-name=1k_retrain
#SBATCH --partition="gpu-medium"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/retrain/HPO_10k/1k_retrain_%A.out
#SBATCH --gres=gpu:1

source activate aizynth-dev
module load CUDA/12.3.0
python start_experiment.py