#!/bin/bash

mkdir -p figures python_outputs
Rscript extract_train_and_test_set.R

python gbm_analysis.py --train-gbm=True --predict-gbm=True
python plot_gbm_results.py
