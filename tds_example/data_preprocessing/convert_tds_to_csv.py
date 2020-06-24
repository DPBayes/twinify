import pandas as pd
import os.path
from urllib.request import urlretrieve
import argparse

parser = argparse.ArgumentParser("script to (down)load data set, select important features and save as csv")
parser.add_argument("--output_dir", default="./", help="path to store output csv files")
args = parser.parse_args()

"""
dataset.xlsx is from https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/data/dataset.xlsx
"""

dataset_path = os.path.join(os.path.dirname(__file__), "dataset.xlsx")
if not os.path.exists(dataset_path):
    print("dataset.xlsx not found locally, downloading....")
    dataset_url = "https://github.com/souzatharsis/covid-19-ML-Lab-Test/raw/master/data/dataset.xlsx"
    urlretrieve(dataset_url, dataset_path)

original_data = pd.read_excel(dataset_path)
feature_names = list(original_data.columns)
feature_names = [name.strip() for name in feature_names]
tds_data = pd.DataFrame(original_data.values, columns=feature_names)

## save top 10 features
top10_features = []
with open(os.path.join(os.path.dirname(__file__), "tds_top10_features.txt"), "r") as top10_features_handle:
    top10_features = top10_features_handle.readlines()

outcome_var = "SARS-Cov-2 exam result"
top10_features.append(outcome_var)

top10_features = [elem.replace("\n", "") for elem in top10_features]

tds_top10 = tds_data[top10_features]
tds_data.to_csv(os.path.join(args.output_dir, "tds_data.csv"), index=False)
tds_top10.to_csv(os.path.join(args.output_dir, "tds_top10.csv"), index=False)

### save also  data set with all features used in TDS example
all_features = []
with open(os.path.join(os.path.dirname(__file__), "tds_all_features.txt"), "r") as all_features_handle:
    all_features = all_features_handle.readlines()

outcome_var = "SARS-Cov-2 exam result"
all_features.append(outcome_var)

all_features = [elem.replace("\n", "") for elem in all_features]

tds_all = tds_data[all_features]
tds_all.to_csv(os.path.join(args.output_dir, "tds_all.csv"), index=False)
