#!/bin/bash
## batch script running twinify to replicate covid19 example data given seed and epsilon

ARGS=$@
OUT_FILE_NAME=`echo $ARGS | sed -r 's/--seed=([0-9]+) --epsilon=(.+)/seed\1_eps\2/'`

mkdir -p ../results/full_model
COMMAND="twinify vi ../covid19_data.csv ../models/model.txt ../results/full_model/syn_data_$OUT_FILE_NAME --num_epochs=200 --num_synthetic=20000 --sampling_ratio=0.01 --k=50 --clipping_threshold=1.0 $ARGS"
echo $COMMAND
$COMMAND
