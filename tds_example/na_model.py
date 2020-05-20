import jax.numpy as np
import jax

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal


from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

from twinify.mixture_model import MixtureModel
import numpyro.distributions as dist
from numpyro.primitives import sample, param, deterministic
from dppp.minibatch import minibatch
import jax.numpy as np

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

k = 50
features = ["Leukocytes", "Eosinophils", "Platelets", "Monocytes", "Inf A H1N1 2009", "Rhinovirus/Enterovirus", "SARS-Cov-2 exam result", "Patient addmited to regular ward (1=yes, 0=no)", "Red blood Cells", "Respiratory Syncytial Virus"]#, "Patient age quantile"]

def model(N, num_obs_total=None):
	pis = sample('pis', dist.Dirichlet(np.ones(k)))
	dists = []

	Leuko_na_prob = sample('Leuko_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Leuko_mus = sample('Leukocytes_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Leuko_sig = sample('Leukocytes_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	dists.append(NAModel(dist.Normal(Leuko_mus, Leuko_sig), Leuko_na_prob))

	Eosi_na_prob = sample('Eosi_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Eosi_mus = sample('Eosinophils_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Eosi_sig = sample('Eosinophils_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	dists.append(NAModel(dist.Normal(Eosi_mus, Eosi_sig), Eosi_na_prob))

	Plate_na_prob = sample('Plate_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Plate_mus = sample('Platelets_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Plate_sig = sample('Platelets_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	dists.append(NAModel(dist.Normal(Plate_mus, Plate_sig), Plate_na_prob))

	Mono_na_prob = sample('Mono_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Mono_mus = sample('Monocytes_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Mono_sig = sample('Monocytes_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	dists.append(NAModel(dist.Normal(Mono_mus, Mono_sig), Mono_na_prob))

	h1n1_na_prob = sample('H1N1_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	h1n1_test_logit = sample('Inf A H1N1 2009', dist.Normal(np.zeros((k,)), np.ones(k,)))
	dists.append(NAModel(dist.Bernoulli(logits=h1n1_test_logit), h1n1_na_prob))

	rhino_na_prob = sample('Rhino_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	rhino_test_logit = sample('Rhinovirus/Enterovirus_logit', dist.Normal(np.zeros((k,)), np.ones(k,)))
	dists.append(NAModel(dist.Bernoulli(logits=rhino_test_logit), rhino_na_prob))

	covid_test_logit = sample("covid_test_logit", dist.Normal(np.zeros((k,)), np.ones(k,)))
	dists.append(dist.Bernoulli(logits=covid_test_logit))

	ward_na_prob = sample('Ward_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	ward_test_logit = sample('Wardvirus/Enterovirus_logit', dist.Normal(np.zeros((k,)), np.ones(k,)))
	dists.append(NAModel(dist.Bernoulli(logits=ward_test_logit), ward_na_prob))

	Red_na_prob = sample('Red_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	Red_mus = sample('RedBlood_mus', dist.Normal(0.*np.ones(k), 1.*np.ones(k)))
	Red_sig = sample('RedBlood_sig', dist.Gamma(2.*np.ones(k), 2.*np.ones(k)))
	dists.append(NAModel(dist.Normal(Red_mus, Red_sig), Red_na_prob))

	rsv_na_prob = sample('RSV_na_prob', dist.Beta(2.*np.ones(k), 2.*np.ones(k)))
	rsv_test_logit = sample('RSVvirus/Enterovirus_logit', dist.Normal(np.zeros((k,)), np.ones(k,)))
	dists.append(NAModel(dist.Bernoulli(logits=rsv_test_logit), rsv_na_prob))

	#age_logits = sample('Age_logit', dist.Normal(np.zeros((k,20)), np.ones(k,20)))
	#dists.append(dist.Categorical(logits=age_logits))

	#feature_dtypes = ["float", "float", "float", "int", "int", "int"]
	#feature_dtypes = ["float", "float", "float", "float"]
	feature_dtypes = ["float"]*len(dists)
	with minibatch(N, num_obs_total):
		 x = sample('x', MixtureModel(dists, feature_dtypes, pis), sample_shape=(N,))

def model_args_map(data, **kwargs):
	return (data.shape[0],), kwargs, {'x':data}
