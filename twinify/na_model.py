import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample

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
		return self.sample_with_intermediates(key, sample_shape)

	def sample_with_intermediates(self, key, sample_shape=()):
		assert(len(sample_shape) == 1)

		vals_rng_key, probs_rng_key = jax.random.split(key, 2)
		z = dist.Bernoulli(probs = self._na_prob).sample(probs_rng_key, sample_shape).astype("int")
		vals = self._base_dist.sample(vals_rng_key, sample_shape=sample_shape)
		vals = np.stack([vals, np.nan*np.ones_like(vals)]).squeeze(1)
		return vals[z, np.arange(vals.shape[1])]
		#return vals*(1.-z)+999.*z, z
