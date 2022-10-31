import jax.numpy as np
import numpyro.distributions as dist
from numpyro.primitives import sample, plate

from twinify.dpvi.modelling import get_feature, sample_combined, NAModel

def preprocess(df):
    """
    Args:
        df (pandas.DataFrame): Data as read from the data set file.
    Returns:
        pandas.DataFrame: Data that will be passed to the model function as argument `x`.
"""
    df = df[["Leukocytes", "Rhinovirus/Enterovirus"]]
    df["Rhinovirus/Enterovirus"] = df["Rhinovirus/Enterovirus"].map({"detected": 1, "not detected": 0})
    return df

def postprocess(df):
    """
    Args:
        df (pandas.DataFrame): Generated synthetic data; same features as output by preprocess
    Returns:
        pandas.DataFrame: Data that will be written to the output file.
    """
    df["Rhinovirus/Enterovirus"] = df["Rhinovirus/Enterovirus"].map(lambda v: "detected" if v == 1 else "not detected")
    return df

def model(x=None, num_obs_total=None):
    """
    Args:
        x (jax.numpy.array): Array holding all features of a single data instance.
        num_obs_total (int): Number of total instances in the data set.
    Samples:
        site `x` similar to input x; array holding all features of a single data instance.
    """
    assert x is None or len(np.shape(x)) == 2
    if x is None:
        N = 1
    else:
        N = np.shape(x)[0]
    if num_obs_total is None:
        num_obs_total = N

    assert isinstance(num_obs_total, int) and num_obs_total > 0
    assert N <= num_obs_total

    leuko_mus = sample('Leukocytes_mus', dist.Normal(0., 1.))
    leuko_sig = sample('Leukocytes_sig', dist.Gamma(2., 2.))
    leuko_dist = dist.Normal(leuko_mus, leuko_sig)

    leuko_na_prob = sample('Leukocytes_na_prob', dist.Beta(1., 1.))
    leuko_na_dist = NAModel(leuko_dist, leuko_na_prob)

    rhino_test_logit = sample('Rhinovirus/Enterovirus_logit', dist.Normal(0., 1.))
    rhino_test_dist = dist.Bernoulli(logits=rhino_test_logit)

    rhino_test_na_prob = sample('Rhinovirus/Enterovirus_na_prob', dist.Beta(1., 1.))
    rhino_test_na_dist = NAModel(rhino_test_dist, rhino_test_na_prob)

    with plate("batch", num_obs_total, N):
        x_leuko = get_feature(x, 0)
        x_rhino = get_feature(x, 1)

        y_leuko = sample('Leukocytes', leuko_na_dist, obs=x_leuko)
        y_rhino = sample('Rhinovirus/Enterovirus', rhino_test_na_dist, obs=x_rhino)
        y = sample_combined(y_leuko, y_rhino)
