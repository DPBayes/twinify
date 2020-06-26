#!/bin/bash
## batch script running twinify to replicate covid19 example data non-privately given seed
## will accept but ignore epsilon parameters

ARGS=$@
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

mkdir -p ../results/full_model_nonprivate
COMMAND="python ./twinify_nonprivate.py ../covid19_data.csv ../models/model.txt ../results/full_model_nonprivate/syn_data_$OUT_FILE_NAME %../models/run_params.txt $ARGS"
echo $COMMAND
$COMMAND
