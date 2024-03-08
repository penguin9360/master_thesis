#!/bin/bash
#SBATCH --job-name=AIZ
#SBATCH --partition="amd-short"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --output=/data1/s2917211/s2917211/aizynthfinder/experiment/slurm_output/slurm-%A_%a.out

# source activate aizynth-dev

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))

smiles_csv="/data1/s2917211/s2917211/aizynthfinder/experiment/splitted/smile_split_2.txt"
config="/data1/s2917211/s2917211/aizynthfinder/experiment/azf_config_new.yml"
output="/data1/s2917211/s2917211/aizynthfinder/experiment/results/xgboost/xgboost_2_result.hdf"
stocks="zinc"
policy="uspto"

aizynthcli --smiles ${smiles_csv} --config ${config} --output ${output} --stocks ${stocks} --policy ${policy}