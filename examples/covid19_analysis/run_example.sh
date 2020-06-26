#!/bin/bash

# download data and extracting relevant features
cd data_preprocessing
python download_dataset.py
Rscript extract_feature_names.R
cd ../

# creating the model.txt file for Twinify
cd models
python create_model_txt.py
cd ../

# executing several runs of Twinify (different privacy levels and seeds)
cd twinify_scripts
bash run_all.sh
cd ../

# train classifiers on synthetic data and plot results
cd results
bash run_analysis.sh
cd ../
