import pandas as pd
import numpy as np

epsilons = [1., 2., 4.]
#read results

basic_auc = pd.read_csv("./r_outputs/basic_auc_full_model.csv", names=epsilons, skiprows=[0])

sens_auc = pd.read_csv("./r_outputs/sens_auc_full_model.csv", names=epsilons, skiprows=[0])
sens_sens = pd.read_csv("./r_outputs/sens_sens_full_model.csv", names=epsilons, skiprows=[0])
sens_spec = pd.read_csv("./r_outputs/sens_spec_full_model.csv", names=epsilons, skiprows=[0])

bal_auc = pd.read_csv("./r_outputs/bal_auc_full_model.csv", names=epsilons, skiprows=[0])
bal_sens = pd.read_csv("./r_outputs/bal_sens_full_model.csv", names=epsilons, skiprows=[0])
bal_spec = pd.read_csv("./r_outputs/bal_spec_full_model.csv", names=epsilons, skiprows=[0])


# nondp baseline
nondp_basic_auc = pd.read_csv("./r_outputs/basic_auc_full_model_nonprivate.csv", index_col=0)

nondp_sens_auc = pd.read_csv("./r_outputs/sens_auc_full_model_nonprivate.csv", index_col=0)
nondp_sens_sens = pd.read_csv("./r_outputs/sens_sens_full_model_nonprivate.csv", index_col=0)
nondp_sens_spec = pd.read_csv("./r_outputs/sens_spec_full_model_nonprivate.csv", index_col=0)

nondp_bal_auc = pd.read_csv("./r_outputs/bal_auc_full_model_nonprivate.csv", index_col=0)
nondp_bal_sens = pd.read_csv("./r_outputs/bal_sens_full_model_nonprivate.csv", index_col=0)
nondp_bal_spec = pd.read_csv("./r_outputs/bal_spec_full_model_nonprivate.csv", index_col=0)


# bar plot for basic AUCs
import matplotlib.pyplot as plt

avg_basic_auc = basic_auc.mean()
std_basic_auc = basic_auc.std()

# orig baseline
orig_basic_auc = pd.read_csv("./r_outputs/original_auc.csv", index_col=0).values[0,0]

# non private baseline
nondp_basic_auc = pd.read_csv("./r_outputs/basic_auc_full_model_nonprivate.csv", index_col=0)

plt.bar(range(len(epsilons)), avg_basic_auc, yerr=std_basic_auc, label="Synthetic data", width=0.5)
xmin, xmax = plt.xlim()
plt.hlines(orig_basic_auc, xmin, xmax, label="Original data", color="red")
plt.hlines(nondp_basic_auc.mean(), xmin, xmax, label="Synthetic data (non private)", color="g")
nondp_basic_auc_std = nondp_basic_auc.std()
plt.hlines(nondp_basic_auc.mean()-nondp_basic_auc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")
plt.hlines(nondp_basic_auc.mean()+nondp_basic_auc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")


plt.xticks(range(len(epsilons)), epsilons)
plt.xlabel(r"$\epsilon$")
plt.ylabel("AUC")
plt.legend(loc="lower right")
plt.title("AUC of ROC curve")

plt.show()
#plt.savefig("./figures/auc_bar_plot.pdf", format="pdf", tight_layout=True)
#plt.close()

## bar plot for high sensitivity AUCs
#import matplotlib.pyplot as plt
#
#avg_sens_auc = sens_auc.mean()
#std_sens_auc = sens_auc.std()
#
## orig baseline
#orig_sens_auc = pd.read_csv("./r_outputs/original_sens_auc.csv", index_col=0).values[0,0]
#
#plt.bar(range(len(epsilons)), avg_sens_auc, yerr=std_sens_auc, label="Synthetic data", width=0.5)
#xmin, xmax = plt.xlim()
#plt.hlines(orig_sens_auc, xmin, xmax, label="Original data")
#plt.xticks(range(len(epsilons)), epsilons)
#plt.xlabel(r"$\epsilon$")
#plt.ylabel("AUC")
#plt.legend(loc="lower right")
#plt.title("AUC of ROC curve for high sensitivity classifier (high resource case)")
#
#plt.show()

## scatter plot for sensitivity threshold optimization
## orig baseline
#orig_sens_sens = pd.read_csv("./r_outputs/original_sens_sens.csv", index_col=0).values[0,0]
#orig_sens_spec = pd.read_csv("./r_outputs/original_sens_spec.csv", index_col=0).values[0,0]
#plt.scatter(orig_sens_sens, orig_sens_spec, color='black', s=20)
#
#for epsilon in epsilons:
#    plt.scatter(sens_sens[epsilon], sens_spec[epsilon], label=epsilon, s=10, alpha=0.5)
#plt.xlabel("Sensitivity")
#plt.ylabel("Specificity")
#plt.legend()
#plt.show()
#
## scatter plot for balance threshold optimization
#
#orig_bal_sens = pd.read_csv("./r_outputs/original_bal_sens.csv", index_col=0).values[0,0]
#orig_bal_spec = pd.read_csv("./r_outputs/original_bal_spec.csv", index_col=0).values[0,0]
#plt.scatter(orig_bal_sens, orig_bal_spec, color='black', s=20)
#for epsilon in epsilons:
#    plt.scatter(bal_sens[epsilon], bal_spec[epsilon], label=epsilon, s=10, alpha=0.5)
#plt.xlabel("Sensitivity")
#plt.ylabel("Specificity")
#plt.legend()
#plt.show()
