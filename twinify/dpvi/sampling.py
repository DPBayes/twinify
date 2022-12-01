# Copyright 2021 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive
from typing import Dict

from twinify.dpvi import ModelFunction, GuideFunction

def sample_synthetic_data(
        model: ModelFunction,
        guide: GuideFunction,
        posterior_params: Dict[str, jnp.ndarray],
        sampling_rng: jnp.ndarray,
        num_parameter_samples: int,
        num_record_samples_per_parameter_sample: int
    ) -> Dict[str, jnp.ndarray]:
    """
    Args:
        model: Model function.
        guide: Guide // variational approximation function.
        posterior_params: Dictionary of learned parameters for the guide/model.
        sampling_rng: JAX RNG key for sampling.
        num_parameter_samples: How many different parameter sets to sample from the guide.
            For each of these a synthetic dataset will be sampled.
        num_record_samples_per_parameter_sample: The number of synthetic data points sampled for each parameter set sample.
    Returns:
        A dictionary of np.arrays of synthetic data samples for each sample site in the model.
        Each of these has shape (num_parameter_samples, num_synthetic_records_per_parameter_sample, *sample_site_shape)
    """

    @jax.jit
    def sample_from_ppd(rng_key):
        """ Samples a single parameter vector and
            num_record_samples_per_parameter_sample based on it.
        """
        parameter_sampling_rng, record_sampling_rng = jax.random.split(rng_key)

        # sample single parameter vector
        posterior_sampler = Predictive(
            guide, params=posterior_params, num_samples=1
        )
        posterior_samples = posterior_sampler(parameter_sampling_rng)
        # models always add a superfluous batch dimensions, squeeze it
        posterior_samples = {k: v.squeeze(0) for k,v in posterior_samples.items()}

        # sample num_record_samples_per_parameter_sample data samples
        ppd_sampler = Predictive(model, posterior_samples, batch_ndims=0)
        per_sample_rngs = jax.random.split(
            record_sampling_rng, num_record_samples_per_parameter_sample
        )
        ppd_samples = jax.vmap(ppd_sampler)(per_sample_rngs)
        # models always add a superfluous batch dimensions, squeeze it
        ppd_samples = {k: v.squeeze(1) for k, v in ppd_samples.items()}

        return ppd_samples

    per_parameter_rngs = jax.random.split(sampling_rng, num_parameter_samples)
    ppd_samples = jax.vmap(sample_from_ppd)(per_parameter_rngs)

    return {site: np.asarray(value) for site, value in ppd_samples.items()}
