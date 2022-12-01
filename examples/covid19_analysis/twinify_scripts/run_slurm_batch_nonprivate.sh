#!/bin/bash
# SPDX-License-Identifier: CC-BY-NC-4.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

## slurm batch script running covid19 example with multiple seeds and values for epsilon
#SBATCH --time 4:00:00
#SBATCH --mem=2G
#SBATCH -c 2
#SBATCH -J twinify_full_npriv
#SBATCH --output=../results/full_model_nonprivate/slurm-%A_%a.out
#SBATCH --array=1-10

echo "The non-private components of the example currently do not work. Sorry."

# module load anaconda3/latest
# source activate ~/.conda/envs/twinify

# n=$SLURM_ARRAY_TASK_ID
# ARGS=`sed "${n}q;d" seeds_and_eps.txt`

# srun ./run_twinify_nonprivate.sh $ARGS
