#!/bin/bash
#SBATCH --job-name=HPO_10k
#SBATCH --partition="gpu-long"
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/hpo/HPO_10k_%A.out
#SBATCH --gres=gpu:1

source activate aizynth-dev
module load CUDA/12.3.0
python parameter_evaluation.py