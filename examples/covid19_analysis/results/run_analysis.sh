#!/bin/bash
# SPDX-License-Identifier: CC-BY-NC-4.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

mkdir -p figures python_outputs
Rscript extract_train_and_test_set.R

python gbm_analysis.py --train-gbm=True --predict-gbm=True
python plot_gbm_results.py
