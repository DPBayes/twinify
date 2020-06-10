import numpy as np
import pandas as pd

data = pd.read_excel('dataset.xlsx')
data = pd.DataFrame(data.values, columns=[name.strip() for name in list(data.columns)])
#data.drop('Patient ID')

# load all features used by R script
with open("tds_all_features.txt", "r") as f:
    all_features = f.readlines()
outcome_var = "SARS-Cov-2 exam result"
all_features.append(outcome_var)
all_features = [elem.replace("\n", "") for elem in all_features]
data = data[all_features]

# map strings
data = data.replace('not_done', np.nan)
data = data.replace('Não Realizado', np.nan)
data.replace('<1000', 500)
data = data.replace('positive', 1.)
data = data.replace('negative', 0.)
data = data.replace('detected', 1.)
data = data.replace('not_detected', 0.)

# removing of all rows with <10 non nan values from negative instances
data_pos = data.loc[data["SARS-Cov-2 exam result"] == 1.]
data_neg = data.loc[data["SARS-Cov-2 exam result"] == 0.]
non_nan_threshold = 10
data_neg = data_neg[len(data.columns) - data.isna().sum(axis=1) >= non_nan_threshold]

# ...and splitting train/test
split_ratio = 2/3
train_data_pos = data_pos.iloc[:int(split_ratio*data_pos.shape[0])]
test_data_pos = data_pos.iloc[int(split_ratio*data_pos.shape[0]):]
train_data_neg = data_neg.iloc[:int(split_ratio*data_neg.shape[0])]
test_data_neg = data_neg.iloc[int(split_ratio*data_neg.shape[0]):]
train_data = pd.concat([train_data_pos, train_data_neg], axis=0)
test_data = pd.concat([test_data_pos, test_data_neg], axis=0)

# impute nan values (replace with mean) -- not good but required by sklearn
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data)
train_data = pd.DataFrame(imp.transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imp.transform(test_data), columns=test_data.columns)

from sklearn.ensemble import GradientBoostingClassifier
train_data_x = train_data.drop(outcome_var, axis=1)
train_data_y = train_data[outcome_var]
test_data_x = test_data.drop(outcome_var, axis=1)
test_data_y = test_data[outcome_var]
gbm = GradientBoostingClassifier(n_estimators=500, subsample=0.8).fit(train_data_x, train_data_y)
pos_class_id = np.squeeze(np.where(gbm.classes_==1.))
test_predictions = np.squeeze(gbm.predict_proba(test_data_x)[:,pos_class_id])

from sklearn.metrics import roc_curve, roc_auc_score
auc_score = roc_auc_score(test_data_y, test_predictions)
fpr, tpr, thresholds = roc_curve(test_data_y, test_predictions)
#sc = gbm.score(

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate / sensitivity')
sens = tpr
spec = 1 - fpr