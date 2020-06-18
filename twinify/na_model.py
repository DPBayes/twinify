import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample

class NAModel(dist.Distribution):

    def __init__(self, base_dist, na_prob=0.5, validate_args=None):
        self._base_dist = base_dist
        self._na_prob = na_prob
        super(NAModel, self).__init__()

    @property
    def base_distribution(self):
        return self._base_dist

    def log_prob(self, value):
        log_probs = dist.Bernoulli(probs = self._na_prob).log_prob(np.isnan(value))
        return log_probs + np.isfinite(value)*self._base_dist.log_prob(np.nan_to_num(value))

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        assert(len(sample_shape) <= 1)

        num_samples = 1
        if len(sample_shape) > 0:
            num_samples = sample_shape[0]

        keys = jax.random.split(key, num_samples)

        @jax.vmap
        def sample_single(single_key):
            vals_rng_key, probs_rng_key = jax.random.split(single_key, 2)
            z = dist.Bernoulli(probs = self._na_prob).sample(probs_rng_key).astype("int")
            vals = self._base_dist.sample(vals_rng_key)
            vals = np.stack([vals, np.nan*np.ones_like(vals)])
            return vals[z]

        vals = sample_single(keys)
        if len(sample_shape) == 0:
            vals = vals.squeeze(0)
        return vals, ()
