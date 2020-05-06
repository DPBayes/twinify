import pandas as pd
import numpy as np
from collections import OrderedDict as od
import os

path = os.path.dirname(__file__)
original_data = pd.read_excel(os.path.join(path, 'clinical_data_einstein.xlsx'))

df = original_data.copy()

features = ['Proteina C reativa mg/dL', 'Lactic Dehydrogenase', 'SARS-Cov-2 exam result', 'Patient age quantile']
admitteds = [feature for feature in df.columns if 'addmited' in feature]
features += admitteds
#df = df[features].dropna()

df = df.rename(columns={'Proteina C reativa mg/dL':'crp', 'Lactic Dehydrogenase':'ldh', 'SARS-Cov-2 exam result':'covid_test', 'Patient age quantile':'age'})
admit_maps = od({name : name.split(' to ')[1].split(' (')[0] for name in admitteds})
df = df.rename(columns=admit_maps)
df['covid_test'] = df['covid_test'].map({'negative':0, 'positive':1})

not_admitted = 1-df[list(admit_maps.values())].values.sum(axis=1)
df['not_admitted'] = not_admitted
severity = np.argmax(df[['not_admitted']+list(admit_maps.values())].values, axis=1)

df['severity'] = severity
df = df.drop(columns=list(admit_maps.values())+["not_admitted"])
