import numpy as np
import pandas as pd
import re
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import argparse

parser = argparse.ArgumentParser("Scripts that train and evaluate GBM models on the synthetic data")
parser.add_argument("--train-gbm", default=True, type=bool, help="train gbm models on synthetic data")
parser.add_argument("--predict-gbm", default=True, type=bool, help="use gbm models to make predictions on synthetic data")
args = parser.parse_args()

########################### Original data preprocessing ##############################
outcome_var = "SARS-Cov-2 exam result"
non_nan_threshold = 10

def encoder(data):
    data = data.replace('not_done', np.nan)
    data = data.replace('Não Realizado', np.nan)
    data.replace('<1000', 500)
    data = data.replace('positive', 1.)
    data = data.replace('negative', 0.)
    data = data.replace('detected', 1.)
    data = data.replace('not_detected', 0.)
    return data

train_data = pd.read_csv("original_train_gbm.csv")
test_data = pd.read_csv("original_test_gbm.csv")
synthetic_data = pd.read_csv("full_model/syn_data_seed{}_eps{}.csv".format(0, 2.0))

name_maps = {}
for feature_name in synthetic_data.columns:
    rred_name = re.sub("\W", ".", feature_name)
    if rred_name in train_data.columns:
        name_maps[rred_name] = feature_name
train_data = encoder(train_data.rename(name_maps, axis="columns"))
test_data = encoder(test_data.rename(name_maps, axis="columns"))
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(data)
train_imp = imp.fit(train_data)
train_data_imputed = pd.DataFrame(train_imp.transform(train_data), columns=train_data.columns)
test_data_imputed = pd.DataFrame(train_imp.transform(test_data), columns=test_data.columns)

########################## Learn GBM on original data ##############################
train_data_x = train_data_imputed.drop(outcome_var, axis=1)
train_data_y = train_data_imputed[outcome_var]
test_data_x = test_data_imputed.drop(outcome_var, axis=1)
test_data_y = test_data_imputed[outcome_var]
gbm_seed = 123
gbm = GradientBoostingClassifier(n_estimators=500, subsample=0.8, random_state=gbm_seed)\
        .fit(train_data_x, train_data_y)
pos_class_id = np.squeeze(np.where(gbm.classes_==1.))
test_predictions = np.squeeze(gbm.predict_proba(test_data_x)[:,pos_class_id])

auc_score = roc_auc_score(test_data_y, test_predictions)
fpr, tpr, thresholds = roc_curve(test_data_y, test_predictions)

# optimize the threshold
train_predictions = np.squeeze(gbm.predict_proba(train_data_x)[:,pos_class_id])
train_fpr, train_tpr, train_thresholds = roc_curve(train_data_y, train_predictions)
train_sens = train_tpr
train_spec = 1 - train_fpr
orig_bal_idx = np.argmax(train_sens+train_spec)
orig_balanced_accuracy = np.mean(1*(test_predictions>train_thresholds[orig_bal_idx])==test_data_y)

pd.DataFrame([[auc_score, orig_balanced_accuracy]], columns=["auc", "bal_acc"])\
        .to_csv("./python_outputs/orig_auc_acc.csv", index=False)

########################## Synthetic data load and preprocess ##############################
if args.train_gbm:
    res_dict = {}
    for eps in [1.0, 2.0, 4.0]:
        res = []
        for seed in range(10):
            print("training gbm model for epsilon {}, seed {}".format(eps, seed))
            syn_data_file = "full_model/syn_data_seed{}_eps{}.csv".format(seed, eps)
            synthetic_data = pd.read_csv(syn_data_file)

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

    pickle.dump(res_dict, open("./python_outputs/imp_gbm_from_python_data_from_R.p", "wb"))

if args.predict_gbm:
    basic_auc_dict = {}
    balanced_sens_dict = {}
    balanced_spec_dict = {}
    balanced_acc_dict = {}
    res_dict = pickle.load(open("./python_outputs/imp_gbm_from_python_data_from_R.p", "rb"))
    for eps in [1.0, 2.0, 4.0]:
        res = res_dict[eps]
        aucs = []
        bal_sens = []
        bal_spec = []
        bal_acc = []
        for seed in range(10):
            print("predicting with gbm model for epsilon {}, seed {}".format(eps, seed))

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
    pd.DataFrame(basic_auc_dict).to_csv("./python_outputs/basic_auc_full_model.csv", index=False)
    pd.DataFrame(balanced_sens_dict).to_csv("./python_outputs/bal_sens_full_model.csv", index=False)
    pd.DataFrame(balanced_spec_dict).to_csv("./python_outputs/bal_spec_full_model.csv", index=False)
    pd.DataFrame(balanced_acc_dict).to_csv("./python_outputs/bal_acc_full_model.csv", index=False)


########################## Synthetic non-private data load and preprocess ##############################
if args.train_gbm:
    res = []
    for seed in range(10):
        print("training gbm model for no-DP baseline, seed {}".format(seed))
        synthetic_nondp_data = pd.read_csv("full_model_nonprivate/syn_data_seed{}_eps2.0.csv".format(seed))

        # map strings
        synthetic_nondp_data = encoder(synthetic_nondp_data)

        # reorder columns to match original
        synthetic_nondp_data = synthetic_nondp_data[train_data.columns]

        # removing of all rows with <10 non nan values from negative instances
        synthetic_nondp_data_pos = synthetic_nondp_data.loc[synthetic_nondp_data["SARS-Cov-2 exam result"] == 1.]
        synthetic_nondp_data_neg = synthetic_nondp_data.loc[synthetic_nondp_data["SARS-Cov-2 exam result"] == 0.]
        synthetic_nondp_data_neg = synthetic_nondp_data_neg.dropna(axis=0, thresh=non_nan_threshold)

        synthetic_nondp_data_train = synthetic_nondp_data_pos.append(synthetic_nondp_data_neg)
        # impute missing data
        from sklearn.impute import SimpleImputer
        synthetic_nondp_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        synthetic_nondp_imp_fit = SimpleImputer(missing_values=np.nan, strategy='mean').fit(synthetic_nondp_data_train)
        synthetic_nondp_data_imputed = pd.DataFrame(synthetic_nondp_imp.fit_transform(synthetic_nondp_data_train),\
                columns=synthetic_nondp_data_train.columns)

        ########################## Learn GBM on synthetic_nondp data ##############################
        synthetic_nondp_data_x = synthetic_nondp_data_imputed.drop(outcome_var, axis=1)
        synthetic_nondp_data_y = synthetic_nondp_data_imputed[outcome_var]
        synthetic_nondp_gbm = GradientBoostingClassifier(n_estimators=500, subsample=0.8, random_state=gbm_seed)\
                .fit(synthetic_nondp_data_x, synthetic_nondp_data_y)

        res.append([synthetic_nondp_imp_fit, synthetic_nondp_gbm])

    pickle.dump(res, open("./python_outputs/nondp_imp_gbm_from_python_data_from_R.p", "wb"))

if args.predict_gbm:
    basic_auc_dict = {}
    balanced_sens_dict = {}
    balanced_spec_dict = {}
    balanced_acc_dict = {}
    res = pickle.load(open("./python_outputs/nondp_imp_gbm_from_python_data_from_R.p", "rb"))

    aucs = []
    bal_sens = []
    bal_spec = []
    bal_acc = []
    for seed in range(10):
        print("predicting with gbm model for no-DP baseline, seed {}".format(seed))
        [synthetic_nondp_imp_fit, synthetic_nondp_gbm] = res[seed]
        synthetic_nondp_data = pd.read_csv("full_model_nonprivate/syn_data_seed{}_eps2.0.csv".format(seed))

        # map strings
        synthetic_nondp_data = encoder(synthetic_nondp_data)

        # reorder columns to match original
        synthetic_nondp_data = synthetic_nondp_data[train_data.columns]

        # removing of all rows with <10 non nan values from negative instances
        synthetic_nondp_data_pos = synthetic_nondp_data.loc[synthetic_nondp_data["SARS-Cov-2 exam result"] == 1.]
        synthetic_nondp_data_neg = synthetic_nondp_data.loc[synthetic_nondp_data["SARS-Cov-2 exam result"] == 0.]
        synthetic_nondp_data_neg = synthetic_nondp_data_neg.dropna(axis=0, thresh=non_nan_threshold)

        synthetic_nondp_data_train = synthetic_nondp_data_pos.append(synthetic_nondp_data_neg)
        # impute missing data
        from sklearn.impute import SimpleImputer
        synthetic_nondp_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        synthetic_nondp_imp_fit = SimpleImputer(missing_values=np.nan, strategy='mean').fit(synthetic_nondp_data_train)
        synthetic_nondp_data_imputed = pd.DataFrame(synthetic_nondp_imp.fit_transform(synthetic_nondp_data_train),\
                columns=synthetic_nondp_data_train.columns)
        imputed_testing_x = pd.DataFrame(synthetic_nondp_imp_fit.transform(test_data[synthetic_nondp_data.columns]), \
                columns=synthetic_nondp_data.columns).drop(outcome_var, axis=1)
        synthetic_nondp_pos_class_id = np.squeeze(np.where(synthetic_nondp_gbm.classes_==1.))
        synthetic_nondp_test_predictions = np.squeeze(synthetic_nondp_gbm\
                .predict_proba(imputed_testing_x)[:,synthetic_nondp_pos_class_id])

        synthetic_nondp_auc_score = roc_auc_score(test_data_y, synthetic_nondp_test_predictions)
        syn_fpr, syn_tpr, syn_thresholds = roc_curve(test_data_y, synthetic_nondp_test_predictions)
        syn_sens = syn_tpr
        syn_spec = 1.-syn_fpr
        bal_idx = np.argmax(syn_sens+syn_spec)
        aucs.append(synthetic_nondp_auc_score)
        bal_sens.append(syn_sens[bal_idx])
        bal_spec.append(syn_spec[bal_idx])
        balanced_accuracy = np.mean(1*(synthetic_nondp_test_predictions>syn_thresholds[bal_idx])==test_data_y)
        bal_acc.append(balanced_accuracy)

    pd.DataFrame(aucs).to_csv("./python_outputs/nondp_basic_auc_full_model.csv", index=False)
    pd.DataFrame(bal_sens).to_csv("./python_outputs/nondp_bal_sens_full_model.csv", index=False)
    pd.DataFrame(bal_spec).to_csv("./python_outputs/nondp_bal_spec_full_model.csv", index=False)
    pd.DataFrame(bal_acc).to_csv("./python_outputs/nondp_bal_acc_full_model.csv", index=False)

print("gbm analysis done")