#!/bin/bash
#SBATCH --job-name=AIZ
#SBATCH --partition="cpu-long"
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=75GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --array=943,966
#SBATCH --output=/data1/s2917211/s2917211/aizynthfinder/experiment/slurm_output/200kValidation4/slurm-%A_%a.out

# source activate aizynth-dev

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))

smiles_csv="/data1/s2917211/s2917211/aizynthfinder/experiment/splitted/200kValidation4/smile_split_${ROW_INDEX}.txt"
config="/data1/s2917211/s2917211/aizynthfinder/experiment/azf_config_200k.yml"
output="/data1/s2917211/s2917211/aizynthfinder/experiment/results/200kValidation4/200kValidation4_${ROW_INDEX}_result.hdf"
stocks="zinc"
policy="uspto"

aizynthcli --smiles ${smiles_csv} --config ${config} --output ${output} --stocks ${stocks} --policy ${policy}