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
    parameter_sampling_rng, record_sampling_rng = jax.random.split(sampling_rng)

    parameter_samples = Predictive(
        guide, params=posterior_params, num_samples=num_parameter_samples
    )(parameter_sampling_rng)

    def _reshape_parameter_value(v: jnp.ndarray) -> jnp.ndarray:
        original_shape = jnp.shape(v)
        assert(original_shape[0] == num_parameter_samples)
        new_shape = (original_shape[0], num_record_samples_per_parameter_sample, *original_shape[1:])
        v = jnp.repeat(v, num_record_samples_per_parameter_sample, axis=0)
        v = jnp.reshape(v, new_shape)
        return v

    parameter_samples = {
        k: _reshape_parameter_value(v)
        for k, v in parameter_samples.items()
    }

    posterior_samples = Predictive(
        model, posterior_samples=parameter_samples, batch_ndims=2
    )(record_sampling_rng)

    return posterior_samples

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
