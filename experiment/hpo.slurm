#!/bin/bash
#SBATCH --job-name=HPO_1k
#SBATCH --partition="gpu-short"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=slurm_output/hpo/HPO_1k_%A.out
#SBATCH --gres=gpu:1

source activate aizynth-dev
module load CUDA/12.3.0
python parameter_evaluation.py