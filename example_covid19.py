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

feature_dtypes = ['float', 'float', 'bool', 'int', 'int']
#crp: Normal, loc=.5


feature_dists = automodel.parse_model(model_str)
feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)

guide_param_sites = automodel.extract_parameter_sites(feature_dists_and_shapes)
guide_dists = automodel.create_guide_dists(guide_param_sites)
guide = automodel.make_guide(guide_dists)
seeded_guide = seed(guide, jax.random.PRNGKey(23496))

prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
model = automodel.make_model(feature_dists_and_shapes, prior_dists, feature_dtypes, k)
seeded_model = seed(model, jax.random.PRNGKey(8365))

from covid19.data.preprocess_einstein import df

profit = train_model_no_dp(
    jax.random.PRNGKey(0),
    model, automodel.model_args_map, guide, automodel.guide_args_map,
    df[list(feature_dists.keys())].to_numpy(),
    batch_size=10,
    num_epochs=100
)
