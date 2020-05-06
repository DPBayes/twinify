from collections import OrderedDict

import jax.numpy as np
import jax

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import sample, param, deterministic
from numpyro.optim import Adam
from numpyro.infer import ELBO, SVI
from dppp.svi import DPSVI
from dppp.modelling import sample_prior_predictive, make_observed_model, sample_multi_posterior_predictive
from dppp.util import unvectorize_shape_2d
from dppp.minibatch import minibatch, subsample_batchify_data
import matplotlib.pyplot as plt
from numpyro.handlers import seed, trace

from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp
import argparse
import datetime
import os.path

k = 7
shapes = {
    'crp': (k,),
    'ldh': (k,),
    'covid_test': (k,),
    'age': (k,),
    'severity': (k, 4)
}

model_str = """
crp: Normal
#ldh: Normal
covid_test: Bernoulli
#age: Poisson
severity: Categorical
"""

#crp: Normal, loc=.5

onp.random.seed(0)

feature_dists = automodel.parse_model(model_str)
feature_dtypes = automodel.parse_support(model_str)
feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)


# def guide(**kwargs):
#     covid_test_probs_loc_uncons = param('covid_test_probs_loc_uncons', .1*onp.random.randn(k))
#     covid_test_probs_std_uncons = param('covid_test_probs_std_uncons', .1*onp.random.randn(k))
#     sample('covid_test_probs', dist.TransformedDistribution(dist.Normal(covid_test_probs_loc_uncons, np.exp(covid_test_probs_std_uncons)), dist.transforms.SigmoidTransform()))

#     crp_loc_loc_uncons = param('crp_loc_loc_uncons', .1*onp.random.randn(k))
#     crp_loc_std_uncons = param('crp_loc_std_uncons', .1*onp.random.randn(k))
#     sample('crp_loc', dist.TransformedDistribution(dist.Normal(crp_loc_loc_uncons, np.exp(crp_loc_std_uncons)), dist.transforms.IdentityTransform()))

#     crp_std_loc_uncons = param('crp_scale_loc_uncons', .1*onp.random.randn(k))
#     crp_std_std_uncons = param('crp_scale_std_uncons', .1*onp.random.randn(k))
#     sample('crp_scale', dist.TransformedDistribution(dist.Normal(crp_std_loc_uncons, np.exp(crp_std_std_uncons)), dist.transforms.ExpTransform()))

guide_param_sites = automodel.extract_parameter_sites(feature_dists_and_shapes)
guide = automodel.make_guide(guide_param_sites, k)
seeded_guide = seed(guide, jax.random.PRNGKey(23496))


prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
model = automodel.make_model(feature_dists_and_shapes, prior_dists, feature_dtypes, k)
seeded_model = seed(model, jax.random.PRNGKey(8365))

from covid19.data.preprocess_einstein import df
train_df = df[list(feature_dists.keys())].dropna()

from numpyro.contrib.autoguide import AutoDiagonalNormal

guide = AutoDiagonalNormal(make_observed_model(model, automodel.model_args_map))

# import pandas as pd
# train_df = pd.DataFrame(onp.stack([onp.random.randn(10000), onp.random.binomial(1, .2, 10000)]).T, columns=['crp', 'covid_test'])

posterior_params = train_model(
    jax.random.PRNGKey(0),
    model, automodel.model_args_map, guide, None,
    train_df.to_numpy(),
    batch_size=100,
    num_epochs=1000,
    dp_scale=2.0
)

posterior_samples = guide.sample_posterior(jax.random.PRNGKey(0), posterior_params, sample_shape=(1000,))

pis = np.mean(posterior_samples['pis'], axis=0)
sev_probs = np.mean(posterior_samples['severity_probs'], axis=0)
sev_probs_marginal = np.sum(pis * sev_probs.T, axis=1)
