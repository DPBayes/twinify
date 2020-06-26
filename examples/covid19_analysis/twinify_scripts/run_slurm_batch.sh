#!/bin/bash
## slurm batch script running covid19 example with multiple seeds and values for epsilon
#SBATCH --time 4:00:00
#SBATCH --mem=2G
#SBATCH -c 2
#SBATCH -J twinify_full
#SBATCH --output=../results/full_model/slurm-%A_%a.out
#SBATCH --array=1-30

module load anaconda3/latest
source activate ~/.conda/envs/twinify

n=$SLURM_ARRAY_TASK_ID
ARGS=`sed "${n}q;d" seeds_and_eps.txt`

srun ./run_twinify.sh $ARGS
