#!/bin/bash
#SBATCH --job-name=HPO_50k
#SBATCH --partition="gpu-long"
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=75GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/hpo/HPO_50k_%A.out
#SBATCH --gres=gpu:a100:1

source activate aizynth-dev
module load CUDA/12.3.0
python parameter_evaluation.py