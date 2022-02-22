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
DP-SVI inference routines used by twinify main script.
"""

from argparse import ArgumentError
import jax.numpy as np
import jax
import pandas as pd

from numpyro.optim import Adam
from numpyro.infer import Trace_ELBO, SVI
from d3p.svi import DPSVI, DPSVIState
from d3p.minibatch import subsample_batchify_data
import d3p.random

class InferenceException(Exception):
    pass

def _cast_data_tuple(data_tuple):
    def _cast_data(df):
        if isinstance(df, (pd.DataFrame, pd.Series)):
            return df.values
        else:
          raise ArgumentError(f"Inference got non pd.DataFrame/pd.Series argument {type(df)}.")

    return tuple(_cast_data(x) for x in data_tuple)


def reset_optimizer(svi: DPSVI, current_state: DPSVIState) -> DPSVIState:
    params = svi.optim.get_params(current_state.optim_state)
    new_optimizer_state = svi.optim.init(params)
    return DPSVIState(new_optimizer_state, current_state.rng_key, current_state.observation_scale)


def _train_model(rng, rng_suite, svi, data, batch_size, num_data, num_epochs, silent=False, reset_optimizer_after=None):
    rng, svi_rng, init_batch_rng = rng_suite.split(rng, 3)

    #init_batching, get_batch = subsample_batchify_data((data,), batch_size)
    assert(type(data) == tuple)
    data = _cast_data_tuple(data)
    init_batching, get_batch = subsample_batchify_data(data, batch_size, rng_suite=rng_suite)
    _, batchify_state = init_batching(init_batch_rng)

    batch = get_batch(0, batchify_state)
    svi_state = svi.init(svi_rng, *batch)

    @jax.jit
    def train_epoch(num_epoch_iter, svi_state, batchify_state):
        def train_iteration(i, state_and_loss):
            svi_state, loss = state_and_loss
            batch = get_batch(i, batchify_state)
            svi_state, iter_loss = svi.update(svi_state, *batch)
            return (svi_state, loss + iter_loss / num_epoch_iter)

        return jax.lax.fori_loop(0, num_epoch_iter, train_iteration, (svi_state, 0.))

    rng, epochs_rng = rng_suite.split(rng)

    for i in range(num_epochs):
        if reset_optimizer_after is not None and i % reset_optimizer_after:
            svi_state = reset_optimizer(svi, svi_state)

        batchify_rng = rng_suite.fold_in(epochs_rng, i)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        if np.isnan(loss):
            raise InferenceException
        loss /= num_data
        if not silent: print("epoch {}: loss {}".format(i, loss))


    return svi.get_params(svi_state), loss

def train_model(rng, rng_suite, model, guide, data, batch_size, num_data, dp_scale, num_epochs, clipping_threshold=1.):
    """ trains a given model using DPSVI and the globally defined parameters and data """

    optimizer = Adam(1e-3)

    svi = DPSVI(
        model, guide, optimizer, Trace_ELBO(),
        num_obs_total=num_data, clipping_threshold=clipping_threshold,
        dp_scale=dp_scale, rng_suite=rng_suite
    )

    return _train_model(rng, rng_suite, svi, data, batch_size, num_data, num_epochs, reset_optimizer_after=100)

def train_model_no_dp(rng, model, guide, data, batch_size, num_data, num_epochs, silent=False, **kwargs):
    """ trains a given model using SVI (no DP!) and the globally defined parameters and data """

    optimizer = Adam(1e-3)

    svi = SVI(
        model, guide,
        optimizer, Trace_ELBO(),
        num_obs_total = num_data
    )

    import d3p.random.debug
    return _train_model(
        d3p.random.convert_to_jax_rng_key(rng),
        d3p.random.debug,
        svi,
        data,
        batch_size,
        num_data,
        num_epochs,
        silent,
        reset_optimizer_after=None
    )
