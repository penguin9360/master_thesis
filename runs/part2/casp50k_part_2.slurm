#!/bin/bash
#SBATCH --job-name=AIZ
#SBATCH --error=%x_%j.err
#SBATCH --partition="cpu-medium"
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5GB
#SBATCH --mail-user="s2917211@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --array=0-599

ROW_INDEX=$((SLURM_ARRAY_TASK_ID+600))

smiles_csv="/home/hassenak/data1/experiments/uspto50k/uspto50k_mcts/splitted/uspto50kMS_AZF_prepro_TEST_RESULTS_part_${ROW_INDEX}.csv"
config="/home/hassenak/data1/experiments/uspto50k/uspto50k_mcts/aizynthfinder_config.yml"
output="/home/hassenak/data1/experiments/uspto50k/uspto50k_mcts/results/uspto50kMS_AZF_prepro_TEST_RESULTS_part_${ROW_INDEX}_result.hdf"
stocks="zinc"
policy="uspto50k"

aizynthcli --smiles ${smiles_csv} --config ${config} --output ${output} --stocks ${stocks} --policy ${policy}