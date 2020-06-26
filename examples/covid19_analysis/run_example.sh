#!/bin/bash

cd data_preprocessing
python download_dataset.py
Rscript extract_feature_names.R
cd ../


cd models
python create_model_txt.py
cd ../

# running twinify here

cd results
bash run_analysis.sh
cd ../