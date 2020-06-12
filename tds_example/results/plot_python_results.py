import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


epsilons = [1., 2., 4.]
#read results
basic_auc = pd.read_csv("./python_outputs/basic_auc_full_model.csv")

bal_acc = pd.read_csv("./python_outputs/bal_acc_full_model.csv")
bal_sens = pd.read_csv("./python_outputs/bal_sens_full_model.csv")
bal_spec = pd.read_csv("./python_outputs/bal_spec_full_model.csv")


# bar plot for basic AUCs
import matplotlib.pyplot as plt

avg_basic_auc = basic_auc.mean()
std_basic_auc = basic_auc.std()

# orig baseline
orig_res = pd.read_csv("./python_outputs/orig_auc_acc.csv")
orig_basic_auc = orig_res["auc"]
orig_bal_acc = orig_res["bal_acc"]

# non private baseline
nondp_basic_auc = pd.read_csv("./python_outputs/nondp_basic_auc_full_model.csv")

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

#plt.show()
plt.savefig("./figures/auc_bar_plot_from_python.pdf", format="pdf", tight_layout=True)
plt.close()


# plot balanced accuracies
avg_bal_acc = bal_acc.mean()
std_bal_acc = bal_acc.std()

# orig baseline
orig_bal_acc = orig_res["bal_acc"]

# non private baseline
nondp_bal_acc = pd.read_csv("./python_outputs/nondp_bal_acc_full_model.csv")

plt.bar(range(len(epsilons)), avg_bal_acc, yerr=std_bal_acc, label="Synthetic data", width=0.5)
xmin, xmax = plt.xlim()
plt.hlines(orig_bal_acc, xmin, xmax, label="Original data", color="red")
plt.hlines(nondp_bal_acc.mean(), xmin, xmax, label="Synthetic data (non private)", color="g")
nondp_bal_acc_std = nondp_bal_acc.std()
plt.hlines(nondp_bal_acc.mean()-nondp_bal_acc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")
plt.hlines(nondp_bal_acc.mean()+nondp_bal_acc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")


plt.xticks(range(len(epsilons)), epsilons)
plt.xlabel(r"$\epsilon$")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title("Classification accuracy (optimized threshold)")

#plt.show()
plt.savefig("./figures/bal_acc_bar_plot_from_python.pdf", format="pdf", tight_layout=True)
plt.close()






#### auc box plot

plt.boxplot(basic_auc.T, labels=epsilons)
xmin, xmax = plt.xlim()
plt.hlines(orig_basic_auc, xmin, xmax, label="Original data", color="red")
plt.hlines(nondp_basic_auc.mean(), xmin, xmax, label="Synthetic data (non private)", color="g")
nondp_basic_auc_std = nondp_basic_auc.std()
plt.hlines(nondp_basic_auc.mean()-nondp_basic_auc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")
plt.hlines(nondp_basic_auc.mean()+nondp_basic_auc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")


plt.xlabel(r"$\epsilon$")
plt.ylabel("AUC")
plt.legend(loc="lower right")
plt.title("AUC of ROC curve")

#plt.show()
plt.savefig("./figures/auc_box_plot_from_python.pdf", format="pdf", tight_layout=True)
plt.close()

#### acc box plot

plt.boxplot(bal_acc.T, labels=epsilons)
xmin, xmax = plt.xlim()
plt.hlines(orig_bal_acc, xmin, xmax, label="Original data", color="red")
plt.hlines(nondp_bal_acc.mean(), xmin, xmax, label="Synthetic data (non private)", color="g")
nondp_bal_acc_std = nondp_bal_acc.std()
plt.hlines(nondp_bal_acc.mean()-nondp_bal_acc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")
plt.hlines(nondp_bal_acc.mean()+nondp_bal_acc_std, xmin, xmax, ls="--", alpha=0.5, lw=1., color="g")


plt.xlabel(r"$\epsilon$")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title("Classification accuracy (optimized threshold)")

#plt.show()
plt.savefig("./figures/bal_acc_box_plot_from_python.pdf", format="pdf", tight_layout=True)
plt.close()
