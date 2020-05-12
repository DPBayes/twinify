import jax.numpy as np
import jax

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal


from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

onp.random.seed(0)

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

features = ["Leukocytes", "Eosinophils", "Rhinovirus/Enterovirus", "SARS-Cov-2 exam result"]
#features = ["Leukocytes", "Rhinovirus/Enterovirus"]
#features = ["Leukocytes", "Eosinophils"]
#features = ["Leukocytes"]
feature_dtypes = ["float", "float", "float", "float"]
#feature_dtypes = ["float", "float"]
#feature_dtypes = ["float"]

import numpyro.distributions as dist

class NAModel(dist.Distribution):

	def __init__(self, base_dist, na_prob=0.5, validate_args=None):
		self._base_dist = base_dist
		self._na_prob = na_prob
		super(NAModel, self).__init__()

	def log_prob(self, value):
		log_na_prob = np.log(self._na_prob)
		log_probs = dist.Bernoulli(probs = self._na_prob).log_prob(np.isnan(value))
		return log_probs + np.isfinite(value)*self._base_dist.log_prob(np.nan_to_num(value))

	def sample(self, key, sample_shape=()):
		return self.sample_with_intermediates(key, sample_shape)[0]

	def sample_with_intermediates(self, key, sample_shape=()):
		assert(len(sample_shape) == 1)

		vals_rng_key, probs_rng_key = jax.random.split(key, 2)
		z = dist.Bernoulli(probs = self._na_prob).sample(probs_rng_key, sample_shape)
		vals = self._base_dist.sample(vals_rng_key, sample_shape=sample_shape)
		return vals*(1.-z)+999.*z, z

k = 100
def model(N, num_obs_total=None):
	pis = sample('pis', dist.Dirichlet(np.ones(k)))

	Leuko_na_prob = sample('Leuko_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Leuko_mus = sample('Leukocytes_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Leuko_sig = sample('Leukocytes_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	#Leuko_dist = dist.Normal(Leuko_mus, Leuko_sig)
	Leuko_dist = NAModel(dist.Normal(Leuko_mus, Leuko_sig), Leuko_na_prob)
	#with minibatch(N, num_obs_total):
	#	 x = sample('x', NAModel(Leuko_dist, Leuko_na_prob), sample_shape=(N,))

	Eosi_na_prob = sample('Eosi_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Eosi_mus = sample('Eosinophils_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Eosi_sig = sample('Eosinophils_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	#Eosi_dist = dist.Normal(Eosi_mus, Eosi_sig)
	Eosi_dist = NAModel(dist.Normal(Eosi_mus, Eosi_sig), Eosi_na_prob)

	rhino_na_prob = sample('Rhino_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	rhino_test_logit = sample('Rhinovirus/Enterovirus_logit', dist.Normal(np.zeros((k,)), np.ones(k,)))
	#rhino_test_dist = dist.Bernoulli(logits=rhino_test_logit)
	rhino_test_dist = NAModel(dist.Bernoulli(logits=rhino_test_logit), rhino_na_prob)

	covid_test_logit = sample("covid_test_logit", dist.Normal(np.zeros((k,)), np.ones(k,)))
	covid_test_dist = dist.Bernoulli(logits=covid_test_logit)

	dists = [Leuko_dist, Eosi_dist, rhino_test_dist, covid_test_dist]
	#dists = [Leuko_dist, rhino_test_dist]
	#dists = [rhino_test_dist]
	with minibatch(N, num_obs_total):
		 x = sample('x', MixtureModel(dists, feature_dtypes, pis), sample_shape=(N,))

def model_args_map(data, **kwargs):
	return (data.shape[0],), kwargs, {'x':data}

guide = AutoDiagonalNormal(make_observed_model(model, model_args_map))
from numpyro.handlers import seed, trace
seeded_model = seed(model, jax.random.PRNGKey(0))
model_trace = trace(seeded_model).get_trace(10)

import pandas as pd
df = pd.read_csv("tds_example/tds_top10.csv")
train_df = df[features]
feature_maps = {name : {value : iterator for iterator, value in enumerate(onp.unique(train_df[name].dropna()))} \
		for name in train_df.columns if train_df[name].dtype=='O'}
for name in feature_maps.keys():
	train_df[name] = train_df[name].map(feature_maps[name])

posterior_params = train_model_no_dp(
	jax.random.PRNGKey(0),
	model, automodel.model_args_map, guide, None,
	train_df.to_numpy(),
	batch_size=100,
	num_epochs=1000
)

posterior_samples = sample_multi_posterior_predictive(jax.random.PRNGKey(1),\
		10000, model, (1,), guide, (), posterior_params)
syn_data = posterior_samples['x']
syn_data = onp.array(syn_data)
syn_data[syn_data == 999.] = onp.nan
