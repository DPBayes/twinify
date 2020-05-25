import numpy as np
import pandas as pd

data = pd.read_csv("../tds_all.csv")

model_str = ""
for column in data.columns:
	feature = data[column]
	if feature.dtype == 'O':
		if len(np.unique(feature.dropna()))==2:
			model_str += "{}: Bernoulli\n".format(column)
		else:
			model_str += "{}: Categorical\n".format(column)
	elif feature.dtype == 'int':
		model_str += "{}: Poisson\n".format(column)
	elif feature.dtype == 'float':
		model_str += "{}: Normal\n".format(column)
handle = open("full_model.txt", "w")
handle.writelines(model_str)
handle.close()
