import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

def logistic_regression(
    datasets, col_to_predict=None, add_constant=True, return_intervals=False, return_results=False, conf_levels=[], weight=1
    ):
    """Logistic regression for multiple synthetic datasets.

    Args:
        datasets (ndarray): (m, n, d) array of m synthetic datasets of shape (n, d).
        col_to_predict (int, optional): The dependent variable column index. Defaults to using the last column.
        add_constant (bool, optional): Whether to add a bias term to the logistic regression. Defaults to True.
        return_intervals (bool, optional): Whether to return confidence intervals. Requires specifying conf_levels. Defaults to False.
        return_results (bool, optional): Whether to return the statsmodels result objects. Defaults to False.
        conf_levels (list, optional): The confidence levels for returned confidence intervals. Defaults to [].
        weight (float, optional): A weight that is used for all datapoints. Defaults to 1.

    Returns:
        tuple: A tuple of results (point estimates, variance estimates, confidence intervals, result objects). 
        The last two items may not be present depending on argument values.
    """
    m, n, d = datasets.shape
    if col_to_predict is None: col_to_predict = d - 1
    feature_cols = set(range(d))
    feature_cols.remove(col_to_predict)
    feature_cols = list(feature_cols)

    num_coefs = d if add_constant else d - 1
    q = np.zeros((m, num_coefs))
    u = np.zeros((m, num_coefs))
    conf_ints = {conf_level: np.zeros((m, num_coefs, 2)) for conf_level in conf_levels}
    result_objs = []
    for i in range(m):
        dataset = datasets[i,:,:]
        feature_array = sm.add_constant(dataset[:, feature_cols], has_constant="add") if add_constant else dataset[:, feature_cols]
        y_array = dataset[:, col_to_predict]

        try:
            syn_model = sm.GLM(y_array, feature_array, family=sm.families.Binomial(), freq_weights=np.repeat(weight, n))
            syn_result = syn_model.fit()

            for conf_level in conf_levels:
                conf_ints[conf_level][i, :, :] = syn_result.conf_int(1 - conf_level)
            if return_results:
                result_objs.append(syn_result)
            q[i, :] = syn_result.params
            u[i, :] = syn_result.bse**2
        except PerfectSeparationError:
            for conf_level in conf_levels:
                conf_ints[conf_level][i, :, :] = np.full((num_coefs, 2), np.nan)
            q[i, :] = np.full(num_coefs, np.nan)
            u[i, :] = np.full(num_coefs, np.nan)

    if return_intervals:
        if return_results:
            return q, u, conf_ints, result_objs
        else:
            return q, u, conf_ints
    else:
        if return_results:
            return q, u, result_objs
        else:
            return q, u

def logistic_regression_regularised(
    datasets, col_to_predict=None, add_constant=True, return_intervals=False, conf_levels=[], weight=1, regularisation=0.00001, n_bootstrap_samples=50
    ):
    """Logistic regression for multiple synthetic datasets.

    Args:
        datasets (ndarray): (m, n, d) array of m synthetic datasets of shape (n, d).
        col_to_predict (int, optional): The dependent variable column index. Defaults to using the last column.
        add_constant (bool, optional): Whether to add a bias term to the logistic regression. Defaults to True.
        return_intervals (bool, optional): Whether to return confidence intervals. Requires specifying conf_levels. Defaults to False.
        conf_levels (list, optional): The confidence levels for returned confidence intervals. Defaults to [].
        weight (float, optional): A weight that is used for all datapoints. Defaults to 1.
        regularisation (float, optional): l2 regularisation multiplier. Defaults to 0.00001.
        n_bootstrap_samples (int, optional): Number of bootstrap samples. Defaults to 100.

    Returns:
        tuple: A tuple of results (point estimates, variance estimates, confidence intervals). 
        The last item may not be present depending on argument values.
    """
    m, n, d = datasets.shape
    if col_to_predict is None: col_to_predict = d - 1
    feature_cols = set(range(d))
    feature_cols.remove(col_to_predict)
    feature_cols = list(feature_cols)

    num_coefs = d if add_constant else d - 1
    q = np.zeros((m, num_coefs))
    u = np.zeros((m, num_coefs))
    conf_ints = {conf_level: np.zeros((m, num_coefs, 2)) for conf_level in conf_levels}
    for i in range(m):
        dataset = datasets[i,:,:]
        feature_array = sm.add_constant(dataset[:, feature_cols], has_constant="add") if add_constant else dataset[:, feature_cols]
        y_array = dataset[:, col_to_predict]

        try:
            bootstrap_samples = np.zeros((n_bootstrap_samples, num_coefs))
            for j in range(n_bootstrap_samples):
                bootstrap_inds = np.random.choice(n, n, replace=True)
                syn_model = sm.GLM(y_array[bootstrap_inds], feature_array[bootstrap_inds, :], family=sm.families.Binomial(), freq_weights=np.repeat(weight, n))
                syn_result = syn_model.fit_regularized(alpha=regularisation, L1_wt=0.00)
                bootstrap_samples[j, :] = syn_result.params

            for conf_level in conf_levels:
                conf_ints[conf_level][i, :, :] = np.quantile(bootstrap_samples, ((1 - conf_level) / 2, 1 - ((1 - conf_level) / 2)), axis=0).transpose()
            q[i, :] = np.mean(bootstrap_samples, axis=0)
            u[i, :] = np.var(bootstrap_samples, axis=0)
        except PerfectSeparationError:
            for conf_level in conf_levels:
                conf_ints[conf_level][i, :, :] = np.full((num_coefs, 2), np.nan)
            q[i, :] = np.full(num_coefs, np.nan)
            u[i, :] = np.full(num_coefs, np.nan)

    if return_intervals:
        return q, u, conf_ints
    else:
        return q, u

