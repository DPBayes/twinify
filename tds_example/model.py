"""
Rhinovirus/Enterovirus: Bernoulli
Leukocytes: Normal
#Inf A H1N1 2009: Bernoulli
Eosinophils: Normal
#Platelets: Normal
#Patient addmited to regular ward (1=yes, 0=no): Bernoulli
#Respiratory Syncytial Virus: Bernoulli
"""

from twinify.mixture_model import MixtureModel
import numpyro.distributions as dist
from numpyro.primitives import sample, param, deterministic
from dppp.minibatch import minibatch
import jax.numpy as np

#features = ["Rhinovirus/Enterovirus", "Leukocytes"]
features = ["Eosinophils"]
feature_dtypes = ["float"]
k = 10

def model(N, num_obs_total=None):
    pis = sample('pis', dist.Dirichlet(np.ones(k)))

    leuko_mus = sample('Leukocytes_mus', dist.Normal(np.zeros((k,)), np.ones((k,))))
    leuko_sig = sample('Leukocytes_sig', dist.Gamma(2.*np.ones((k,)), 2.*np.ones((k,))))
    leuko_dist = dist.Normal(leuko_mus, leuko_sig)

    #rhino_test_logit = sample('Rhinovirus/Enterovirus_logit', dist.Normal(np.zeros((k,)), np.ones(k,)))
    #rhino_test_dist = dist.Bernoulli(logits=rhino_test_logit)

    dists = [leuko_dist]#, rhino_test_dist]
    with minibatch(N, num_obs_total):
        x = sample('x', MixtureModel(dists, feature_dtypes, pis), sample_shape=(N,))

def model_args_map(data, **kwargs):
    return (data.shape[0],), kwargs, {'x':data}
