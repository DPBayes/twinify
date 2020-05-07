import pandas as pd

"""
dataset.xlsx is from https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/data/dataset.xlsx
"""

tds_data = pd.read_excel("dataset.xlsx")                                                                       
top10_features_handle = open("tds_top10_features.txt", "r")
top10_features = top10_features_handle.readlines()
top10_features_handle.close()

top10_features = [elem.replace("\n", "") for elem in top10_features]

tds_top10 = tds_data[top10_features]
tds_data.to_csv("tds_data.csv", index=False)
tds_top10.to_csv("tds_top10.csv", index=False)
