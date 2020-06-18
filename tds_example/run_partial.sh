#!/bin/bash
## triton batch script running tds example with multiple seeds and values for epsilon
#SBATCH --time 1:00:00
#SBATCH --mem=2G
#SBATCH -c 2
#SBATCH -J twinify_partial
#SBATCH --output=./results/partial_model/slurm-%A_%a.out

module load anaconda3/latest
source activate ~/.conda/envs/twinify

n=$SLURM_ARRAY_TASK_ID
ARGS=`sed "${n}q;d" run_params.txt`
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

mkdir -p ./results/partial_model/
COMMAND="python ../ui.py ./tds_all.csv ./models/partial_model.txt ./results/partial_model/syn_data_$OUT_FILE_NAME %./models/run_params.txt $ARGS"
echo $COMMAND
srun $COMMAND