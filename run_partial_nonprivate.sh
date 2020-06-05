#!/bin/bash
## triton batch script running tds example with multiple seeds and values for epsilon
#SBATCH --time 1:00:00
#SBATCH --mem=2G
#SBATCH -c 2
#SBATCH -J twinify_partial_npriv


module load anaconda3/latest
source activate ~/.conda/envs/twinify

n=$SLURM_ARRAY_TASK_ID
ARGS=`sed "${n}q;d" run_params.txt`
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

COMMAND="python ui.py ./tds_example/tds_all.csv ./tds_example/models/partial_model.txt ./results/partial_model_nonprivate/syn_data_$OUT_FILE_NAME %./tds_example/models/run_params.txt $ARGS"
echo $COMMAND
srun $COMMAND
