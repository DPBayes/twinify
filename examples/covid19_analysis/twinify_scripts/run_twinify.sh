#!/bin/bash
## batch script running twinify to replicate covid19 example data given seed and epsilon

ARGS=$@
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

mkdir -p ../results/full_model
COMMAND="python ../../../twinify/__main__.py ../covid19_data.csv ../models/model.txt ../results/full_model/syn_data_$OUT_FILE_NAME %../models/run_params.txt $ARGS"
echo $COMMAND
$COMMAND
