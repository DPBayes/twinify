import pandas as pd

"""
dataset.xlsx is from https://github.com/souzatharsis/covid-19-ML-Lab-Test/blob/master/data/dataset.xlsx
"""

tds_data = pd.read_excel("dataset.xlsx")                                                                       
top10_features = list(pd.read_csv("tds_top10_features.csv")['x'])
top10_features = [elem.replace(".", " ") for elem in top10_features]
top10_features = ['Rhinovirus/Enterovirus',
 'Leukocytes',
 'Inf A H1N1 2009',
 'Eosinophils',
 'Platelets',
 'Monocytes',
 'Patient addmited to regular ward (1=yes, 0=no)',
 'Red blood Cells',
 'Respiratory Syncytial Virus',
 'Patient age quantile']

tds_top10 = tds_data[top10_features]
tds_data.to_csv("tds_data.csv", index=False)
tds_top10.to_csv("tds_top10.csv", index=False)

