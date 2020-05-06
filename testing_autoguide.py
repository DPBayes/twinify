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

from twinify.infer import train_model, train_model_no_dp, _train_model
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


feature_dists = automodel.parse_model(model_str)
feature_dtypes = automodel.parse_support(model_str)
feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)

prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
model = automodel.make_model(feature_dists_and_shapes, prior_dists, feature_dtypes, k)
#seeded_model = seed(model, jax.random.PRNGKey(8365))


from covid19.data.preprocess_einstein import df
data = df[list(feature_dists.keys())].dropna().to_numpy()													   


import dppp
model_for_numpyro = make_observed_model(model, automodel.model_args_map)												  
#seeded_model = seed(model_for_numpyro, jax.random.PRNGKey(1))				
#print(trace(seeded_model).get_trace(data))

from numpyro.contrib.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model_for_numpyro)
#seeded_guide = seed(guide, jax.random.PRNGKey(23496))


svi = SVI(
	model_for_numpyro, guide,
	Adam(1e-3), ELBO(),
	num_obs_total=data.shape[0]
)

batch_size = 10
num_epochs = 10

asd = _train_model(jax.random.PRNGKey(0), svi, data, 100, 100)


assert(False)
from covid19.data.preprocess_einstein import df
profit = train_model_no_dp(
	jax.random.PRNGKey(0),
	model, automodel.model_args_map, guide, automodel.guide_args_map,
	df[list(feature_dists.keys())].dropna().to_numpy(),
	batch_size=10,
	num_epochs=100
)
