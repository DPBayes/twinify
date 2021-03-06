# Copyright 2020 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Missing value model used by twinify main script and available for modelling.
"""

import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample

class NAModel(dist.Distribution):
    """ Model decorator for missing values.

    Can be used to enhance any single feature distribution to handle missing
    values (represented by nan values (`np.nan`)) in a missing completely at random fashion.
    """

    def __init__(self, base_dist, na_prob=0.5, validate_args=None):
        """
        Initialize a new NAModel instance.

        Args:
            base_dist (numpyro.distributions.Distribution): The feature distribution for existing data.
            na_prob (float): Probability that data is missing.
        """
        self._base_dist = base_dist
        self._na_prob = na_prob
        super(NAModel, self).__init__()

    @property
    def base_distribution(self):
        """ The feature base distribution for existing data. """
        return self._base_dist

    def log_prob(self, value):
        """
        Evaluates the log-probabilities for given data realizations (observed or missing)

        Args:
            values (array_like): The data.
        Returns:
            array_like containing probabilities for each given data point
        """
        log_probs = dist.Bernoulli(probs = self._na_prob).log_prob(np.isnan(value))
        return log_probs + np.isfinite(value)*self._base_dist.log_prob(np.nan_to_num(value))

    def sample(self, key, sample_shape=()):
        """
        Samples from the missing values model.

        Args:
            key (jax.random.PRNGKey): RNG key.
            sample_shape (tuple): How many samples to draw and how to arrange them
        Returns:
            array_like
        """
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
            orig_shape = vals.shape
            if len(vals.shape) == 0:
                vals = vals.reshape((1,))
            vals = np.stack([vals, np.nan*np.ones_like(vals)])
            assert(len(vals.shape) >= 2)
            vals = vals[z, np.arange(vals.shape[1])]
            assert(len(vals.shape) == len(orig_shape) or (len(vals.shape) == len(orig_shape) + 1 and vals.shape[0] == 1))
            vals = vals.reshape(orig_shape)
            return vals

        vals = sample_single(keys)
        if len(sample_shape) == 0:
            vals = vals.squeeze(0)
        return vals, ()
