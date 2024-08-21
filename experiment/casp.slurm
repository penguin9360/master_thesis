#!/bin/bash
#SBATCH --job-name=AIZ
#SBATCH --partition="cpu-medium"
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=75GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --array=54,87,100,106,119,231,239,249,277,298,301-302,354,359,486,511-512,530,545,601,703,718,780,796,899,992
#SBATCH --output=/data1/s2917211/s2917211/aizynthfinder/experiment/slurm_output/50k/slurm-%A_%a.out

# source activate aizynth-dev

ROW_INDEX=$((SLURM_ARRAY_TASK_ID))

smiles_csv="/data1/s2917211/s2917211/aizynthfinder/experiment/splitted/50k/smile_split_${ROW_INDEX}.txt"
config="/data1/s2917211/s2917211/aizynthfinder/experiment/azf_config_200k.yml"
output="/data1/s2917211/s2917211/aizynthfinder/experiment/results/50k/50k_${ROW_INDEX}_result.hdf"
stocks="zinc"
policy="uspto"

aizynthcli --smiles ${smiles_csv} --config ${config} --output ${output} --stocks ${stocks} --policy ${policy}