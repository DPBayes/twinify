#!/bin/bash
## slurm batch script running tds example with multiple seeds and values for epsilon
#SBATCH --time 4:00:00
#SBATCH --mem=2G
#SBATCH -c 4
#SBATCH -J twinify_full
#SBATCH --output=../results/full_model/slurm-%A_%a.out

module load anaconda3/latest
source activate ~/.conda/envs/twinify

n=$SLURM_ARRAY_TASK_ID
ARGS=`sed "${n}q;d" seeds_and_eps.txt`
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

mkdir -p results/full_model
COMMAND="python ../../../twinify/__main__.py ../tds_all.csv ../models/full_model.txt ../results/full_model/syn_data_$OUT_FILE_NAME %../models/run_params.txt $ARGS"
echo $COMMAND
srun $COMMAND
