""" Parses the list of feature names and builds an automodel txt file for twinify.

For this we use a simple heuristic to determine the feature distributions based on the data:
    - Bernoulli distribution if the type of the feature is integer or object and there are 2 possible values
    - Categorical distribution if the type of the feature is object
    - Poisson distribution if the type of the feature is integer
    - Normal distribution for all other cases.
 """

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
