import pandas as pd

"""
dataset.xlsx is from https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/data/dataset.xlsx
"""

tds_data = pd.read_excel("dataset.xlsx") 
feature_names = list(tds_data.columns)
feature_names = [name.strip() for name in feature_names]
tds_data = pd.DataFrame(tds_data, columns=feature_names)

## save top 10 features
top10_features_handle = open("tds_top10_features.txt", "r")
top10_features = top10_features_handle.readlines()
top10_features_handle.close()

outcome_var = "SARS-Cov-2 exam result"
top10_features.append(outcome_var)

top10_features = [elem.replace("\n", "") for elem in top10_features]

tds_top10 = tds_data[top10_features]
tds_data.to_csv("tds_data.csv", index=False)
tds_top10.to_csv("tds_top10.csv", index=False)

### save also  data set with all features used in TDS example
all_features_handle = open("tds_all_features.txt", "r")
all_features = all_features_handle.readlines()
all_features_handle.close()

outcome_var = "SARS-Cov-2 exam result"
all_features.append(outcome_var)

all_features = [elem.replace("\n", "") for elem in all_features]

tds_all = tds_data[all_features]
tds_all.to_csv("tds_all.csv", index=False)
