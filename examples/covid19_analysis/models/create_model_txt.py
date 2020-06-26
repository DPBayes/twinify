import numpy as np
import pandas as pd

with open("../data_preprocessing/covid19_features.txt", "r") as f:
    features = [feature.strip() for feature in f.readlines()]

data = pd.read_csv("../covid19_data.csv")[features]

model_str = ""
for column in data.columns:
    feature = data[column]
    if feature.dtype == 'O':
        if len(np.unique(feature.dropna()))==2:
            model_str += "{}: Bernoulli\n".format(column)
        else:
            model_str += "{}: Categorical\n".format(column)
    elif feature.dtype == 'int':
        if len(np.unique(feature.dropna()))==2:
            model_str += "{}: Bernoulli\n".format(column)
        else:
            model_str += "{}: Poisson\n".format(column)
    elif feature.dtype == 'float':
        model_str += "{}: Normal\n".format(column)

with open("model.txt", "w") as handle:
    handle.writelines(model_str)
