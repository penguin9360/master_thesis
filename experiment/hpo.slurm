#!/bin/bash
#SBATCH --job-name=AIZ
#SBATCH --partition="gpu-medium"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=75GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=/data1/s2917211/s2917211/aizynthfinder/experiment/slurm_output/hpo/hpo-%A_%a.out
#SBATCH --gres=gpu:1 

source activate aizynth-dev
module load CUDA/12.3.0
python parameter_evaluation.py