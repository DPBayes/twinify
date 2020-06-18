import jax.numpy as np
import jax

import numpyro
import numpyro.distributions as dist

################ General mixture model attempt
from jax.scipy.special import logsumexp
import numpyro.distributions as dist

class MixtureModel(dist.Distribution):
    arg_constraints = {
        '_pis' : dist.constraints.simplex
    }

    def __init__(self, dists, pis=1.0, validate_args=None):
        self.dists = dists
        self._pis = pis
        super(MixtureModel, self).__init__()

    def log_prob(self, value):
        log_pis = np.log(self._pis)
        if value.ndim == 2:
            log_phis = np.array([dbn.log_prob(value[:, feat_idx, np.newaxis]) for feat_idx, dbn in \
                    enumerate(self.dists)]).sum(axis=0)
            assert(value.ndim == 2)
            assert(log_phis.shape == (value.shape[0], len(log_pis)))
            temp = log_pis + log_phis[:,:len(log_phis)]
        else:
            log_phis = np.array([dbn.log_prob(value[feat_idx, np.newaxis]) for feat_idx, dbn in \
                    enumerate(self.dists)]).sum(axis=0)
            assert(value.ndim == 1)
            assert(log_phis.shape == (len(log_pis),))
            temp = log_pis + log_phis[:len(log_phis)]

        return logsumexp(temp, axis=-1)

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
            vals_rng_key, pis_rng_key = jax.random.split(single_key, 2)
            z = dist.Categorical(self._pis).sample(pis_rng_key)
            rng_keys = jax.random.split(vals_rng_key, len(self.dists))
            vals = [dbn.sample(rng_keys[feat_idx])[z] \
                    for feat_idx, dbn in enumerate(self.dists)]
            return np.stack(vals).T, z

        vals, zs = sample_single(keys)
        if len(sample_shape) == 0:
            vals = vals.squeeze(0)
        return vals, [zs]
