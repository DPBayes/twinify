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
from typing import Dict, Iterable, Callable, Tuple

from twinify.model_loading import TModelFunction, TGuideFunction

def sample_synthetic_data(
        model: TModelFunction,
        guide: TGuideFunction,
        posterior_params: Dict[str, jnp.ndarray],
        sampling_rng: jnp.ndarray,
        num_parameter_samples: int,
        num_record_samples_per_parameter_sample: int
    ) -> Dict[str, jnp.ndarray]:
    """
    Returns a dictionary jnp.arrays of synthetic data samples for each sample site in the model.
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

    return ppd_samples

PreparedPostprocessFunction = Callable[[Dict[str, jnp.ndarray]], pd.DataFrame]

def reshape_and_postprocess_synthetic_data(
        posterior_samples: Dict[str, jnp.ndarray],
        prepared_postprocess_fn: PreparedPostprocessFunction,
        separate_output: bool,
        num_parameter_samples: int
    ) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    """ Given a dictionary of posterior predictive samples as output by sample_synthetic_data,
    organises it into separate-per-parameter-sample or a single large data set(s) and
    performs post-processing.

    If the `separate_output` argument is `True`, the result will be a list of
    pandas data frames, each of which holding records sampled from a single sample
    from the parameter posterior. If `separate_output` is `False`, the result
    will only contain a single data frame, containing all samples.

    Postprocessing is performed to transform the data used internally by numpyro so
    that the synthetic twin looks like the original data, using either a user-provided
    `postprocess` function or one set-up automatically in the case of automodelling.
    """

    def _squash_sample_dims(v: jnp.array) -> jnp.array:
        old_shape = jnp.shape(v)
        assert len(old_shape) >= 2
        new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
        reshaped_v = jnp.reshape(v, new_shape)
        return reshaped_v

    posterior_samples_array = []
    if separate_output:
        for i in range(num_parameter_samples):
            site_dict = {
                k: v[i] for k, v in posterior_samples.items()
            }
            posterior_samples_array.append(site_dict)
    else:
        posterior_samples_array = [{
            k: _squash_sample_dims(v) for k, v in posterior_samples.items()
        }]

    for posterior_samples in posterior_samples_array:
        syn_df, encoded_syn_df = prepared_postprocess_fn(posterior_samples)
        yield (syn_df, encoded_syn_df)
