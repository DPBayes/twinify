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
    #def __init__(self, dists, pis_unc=.0, validate_args=None):
        self.dists = dists
        self._pis = pis
        #self._pis_unc = pis_unc
        super(MixtureModel, self).__init__()

    def log_prob(self, value):
        log_pis = np.log(self._pis)
        try:
            log_phis = np.array([dbn.log_prob(value[:, feat_idx, np.newaxis]) for feat_idx, dbn in \
                    enumerate(self.dists)]).sum(axis=0)
        except:
            log_phis = np.array([dbn.log_prob(value[feat_idx, np.newaxis]) for feat_idx, dbn in \
                    enumerate(self.dists)]).sum(axis=0)
        temp = log_pis + log_phis
        #print(self._pis_unc.shape)
        #temp = (self._pis_unc-logsumexp(self._pis_unc, axis=-1)) + log_phis

        return logsumexp(temp, axis=-1)

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        assert(len(sample_shape) == 1)

        vals_rng_key, pis_rng_key = jax.random.split(key, 2)
        z = dist.Categorical(self._pis).sample(pis_rng_key, sample_shape)
        rng_keys = jax.random.split(vals_rng_key, len(self.dists))
        vals = [dbn.sample(rng_keys[feat_idx], sample_shape=sample_shape)[np.arange(sample_shape[0]), z] \
                 for feat_idx, dbn in enumerate(self.dists)]
        return np.stack(vals).squeeze(-1).T, [z]
