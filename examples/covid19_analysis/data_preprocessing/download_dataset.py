# SPDX-License-Identifier: CC-BY-NC-4.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

import pandas as pd
import os.path
from urllib.request import urlretrieve
import argparse

parser = argparse.ArgumentParser("script to (down)load data set, select important features and save as csv")

"""
dataset.xlsx is from https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/data/dataset.xlsx
"""

workdir_path = os.path.join(os.path.dirname(__file__), "..")
dataset_path = os.path.join(workdir_path, "dataset.xlsx")
if not os.path.exists(dataset_path):
    print("dataset.xlsx not found locally, downloading....")
    dataset_url = "https://github.com/souzatharsis/covid-19-ML-Lab-Test/raw/master/data/dataset.xlsx"
    urlretrieve(dataset_url, dataset_path)

original_data = pd.read_excel(dataset_path)
feature_names = list(original_data.columns)
feature_names = [name.strip() for name in feature_names]
covid19_data = pd.DataFrame(original_data.values, columns=feature_names)

covid19_data.to_csv(os.path.join(workdir_path, "covid19_data.csv"), index=False)
