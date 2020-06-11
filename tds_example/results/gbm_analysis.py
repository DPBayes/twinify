import numpy as np
import pandas as pd

data = pd.read_excel('../dataset.xlsx')
data = pd.DataFrame(data.values, columns=[name.strip() for name in list(data.columns)])
del data["Patient ID"]
########################### Original data preprocessing ##############################
outcome_var = "SARS-Cov-2 exam result"

def encoder(data):
    data = data.replace('not_done', np.nan)
    data = data.replace('Não Realizado', np.nan)
    data.replace('<1000', 500)
    data = data.replace('positive', 1.)
    data = data.replace('negative', 0.)
    data = data.replace('detected', 1.)
    data = data.replace('not_detected', 0.)
    return data

# encode features
data = encoder(data) 

## remove features that have >95% missing values
N = len(data)
not_na_pct = 0.05
data = data.dropna(axis=1, thresh=N*not_na_pct)

## removing of all rows with <10 non nan values from negative instances
data_pos = data.loc[data["SARS-Cov-2 exam result"] == 1.]
data_neg = data.loc[data["SARS-Cov-2 exam result"] == 0.]
non_nan_threshold = 10
data_neg = data_neg[len(data.columns) - data.isna().sum(axis=1) >= non_nan_threshold]

## ...and splitting train/test
split_ratio = 2/3
train_data_pos = data_pos.iloc[:int(split_ratio*data_pos.shape[0])]
test_data_pos = data_pos.iloc[int(split_ratio*data_pos.shape[0]):]
train_data_neg = data_neg.iloc[:int(split_ratio*data_neg.shape[0])]
test_data_neg = data_neg.iloc[int(split_ratio*data_neg.shape[0]):]
train_data = pd.concat([train_data_pos, train_data_neg], axis=0)
test_data = pd.concat([test_data_pos, test_data_neg], axis=0)

## impute nan values (replace with mean) -- not good but required by sklearn
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_imp = imp.fit(train_data)
train_data_imputed = pd.DataFrame(train_imp.transform(train_data), columns=train_data.columns)
test_data_imputed = pd.DataFrame(train_imp.transform(test_data), columns=test_data.columns)

########################## Learn GBM on original data ##############################
from sklearn.ensemble import GradientBoostingClassifier
train_data_x = train_data_imputed.drop(outcome_var, axis=1)
train_data_y = train_data_imputed[outcome_var]
test_data_x = test_data_imputed.drop(outcome_var, axis=1)
test_data_y = test_data_imputed[outcome_var]
gbm_seed = 123
gbm = GradientBoostingClassifier(n_estimators=500, subsample=0.8, random_state=gbm_seed)\
        .fit(train_data_x, train_data_y)
pos_class_id = np.squeeze(np.where(gbm.classes_==1.))
test_predictions = np.squeeze(gbm.predict_proba(test_data_x)[:,pos_class_id])

from sklearn.metrics import roc_curve, roc_auc_score
auc_score = roc_auc_score(test_data_y, test_predictions)
fpr, tpr, thresholds = roc_curve(test_data_y, test_predictions)

# optimize the threshold
train_predictions = np.squeeze(gbm.predict_proba(train_data_x)[:,pos_class_id])
train_fpr, train_tpr, train_thresholds = roc_curve(train_data_y, train_predictions)
train_sens = train_tpr
train_spec = 1 - train_fpr
orig_bal_idx = np.argmax(train_sens+train_spec)
orig_balanced_accuracy = np.mean(1*(test_predictions>train_thresholds[orig_bal_idx])==test_data_y)

########################## Synthetic data load and preprocess ##############################
train = False
import pickle
if train:
    res_dict = {}
    for eps in [1.0, 2.0, 4.0]:
        res = []
        for seed in range(10):
            synthetic_data = pd.read_csv("full_model/syn_data_seed{}_eps{}.csv".format(seed, eps))

            # map strings
            synthetic_data = encoder(synthetic_data)

            # reorder columns to match original
            synthetic_data = synthetic_data[train_data.columns]

            # removing of all rows with <10 non nan values from negative instances
            synthetic_data_pos = synthetic_data.loc[synthetic_data["SARS-Cov-2 exam result"] == 1.]
            synthetic_data_neg = synthetic_data.loc[synthetic_data["SARS-Cov-2 exam result"] == 0.]
            synthetic_data_neg = synthetic_data_neg.dropna(axis=0, thresh=non_nan_threshold)

            synthetic_data_train = synthetic_data_pos.append(synthetic_data_neg)
            # impute missing data
            from sklearn.impute import SimpleImputer
            synthetic_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            synthetic_imp_fit = SimpleImputer(missing_values=np.nan, strategy='mean').fit(synthetic_data_train)
            synthetic_data_imputed = pd.DataFrame(synthetic_imp.fit_transform(synthetic_data_train),\
                    columns=synthetic_data_train.columns)

            ########################## Learn GBM on synthetic data ##############################
            synthetic_data_x = synthetic_data_imputed.drop(outcome_var, axis=1)
            synthetic_data_y = synthetic_data_imputed[outcome_var]
            synthetic_gbm = GradientBoostingClassifier(n_estimators=500, subsample=0.8, random_state=gbm_seed)\
                    .fit(synthetic_data_x, synthetic_data_y)
            synthetic_pos_class_id = np.squeeze(np.where(synthetic_gbm.classes_==1.))
            imputed_testing_x = pd.DataFrame(synthetic_imp_fit.transform(test_data[synthetic_data.columns]), \
                    columns=synthetic_data.columns).drop(outcome_var, axis=1)
            synthetic_test_predictions = np.squeeze(synthetic_gbm\
                    .predict_proba(imputed_testing_x)[:,synthetic_pos_class_id])

            synthetic_auc_score = roc_auc_score(test_data_y, synthetic_test_predictions)
            syn_fpr, syn_tpr, syn_thresholds = roc_curve(test_data_y, synthetic_test_predictions)
            syn_sens = syn_tpr
            syn_spec = 1.-syn_fpr
            res.append([synthetic_imp_fit, synthetic_gbm])
        res_dict[eps] = res

    pickle.dump(res_dict, open("imp_gbm_from_python.p", "wb"))

else:
    basic_auc_dict = {}
    balanced_sens_dict = {}
    balanced_spec_dict = {}
    balanced_acc_dict = {}
    res_dict = pickle.load(open("imp_gbm_from_python.p", "rb"))
    for eps in [1.0, 2.0, 4.0]:
        res = res_dict[eps]
        aucs = []
        bal_sens = []
        bal_spec = []
        bal_acc = []
        for seed in range(10):
            [synthetic_imp_fit, synthetic_gbm] = res[seed]
            synthetic_data = pd.read_csv("full_model/syn_data_seed{}_eps{}.csv".format(seed, eps))

            # map strings
            synthetic_data = encoder(synthetic_data)

            # reorder columns to match original
            synthetic_data = synthetic_data[train_data.columns]

            # removing of all rows with <10 non nan values from negative instances
            synthetic_data_pos = synthetic_data.loc[synthetic_data["SARS-Cov-2 exam result"] == 1.]
            synthetic_data_neg = synthetic_data.loc[synthetic_data["SARS-Cov-2 exam result"] == 0.]
            synthetic_data_neg = synthetic_data_neg.dropna(axis=0, thresh=non_nan_threshold)

            synthetic_data_train = synthetic_data_pos.append(synthetic_data_neg)
            # impute missing data
            from sklearn.impute import SimpleImputer
            synthetic_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            synthetic_imp_fit = SimpleImputer(missing_values=np.nan, strategy='mean').fit(synthetic_data_train)
            synthetic_data_imputed = pd.DataFrame(synthetic_imp.fit_transform(synthetic_data_train),\
                    columns=synthetic_data_train.columns)
            imputed_testing_x = pd.DataFrame(synthetic_imp_fit.transform(test_data[synthetic_data.columns]), \
                    columns=synthetic_data.columns).drop(outcome_var, axis=1)
            synthetic_pos_class_id = np.squeeze(np.where(synthetic_gbm.classes_==1.))
            synthetic_test_predictions = np.squeeze(synthetic_gbm\
                    .predict_proba(imputed_testing_x)[:,synthetic_pos_class_id])

            synthetic_auc_score = roc_auc_score(test_data_y, synthetic_test_predictions)
            print(synthetic_auc_score)
            syn_fpr, syn_tpr, syn_thresholds = roc_curve(test_data_y, synthetic_test_predictions)
            syn_sens = syn_tpr
            syn_spec = 1.-syn_fpr
            bal_idx = np.argmax(syn_sens+syn_spec)
            aucs.append(synthetic_auc_score)
            bal_sens.append(syn_sens[bal_idx])
            bal_spec.append(syn_spec[bal_idx])
            balanced_accuracy = np.mean(1*(synthetic_test_predictions>syn_thresholds[bal_idx])==test_data_y)
            bal_acc.append(balanced_accuracy)
        basic_auc_dict[eps] = aucs
        balanced_sens_dict[eps] = bal_sens
        balanced_spec_dict[eps] = bal_spec
        balanced_acc_dict[eps] = bal_acc
