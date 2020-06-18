import jax.numpy as np
import jax

import numpyro
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import ELBO, SVI
from dppp.svi import DPSVI
from dppp.modelling import make_observed_model
from dppp.minibatch import minibatch, subsample_batchify_data

from numpyro.infer.svi import SVIState

class InferenceException(Exception):
    pass

def _train_model(rng, svi, data, batch_size, num_epochs):
    rng, svi_rng, init_batch_rng = jax.random.split(rng, 3)

    init_batching, get_batch = subsample_batchify_data((data,), batch_size)
    _, batchify_state = init_batching(init_batch_rng)

    batch = get_batch(0, batchify_state)
    svi_state = svi.init(svi_rng, *batch)

    ##### hotfix to prevent double jit compilation #####
    ###### remove this once fixed in numpyro/jax #######
    optim_state = svi_state.optim_state
    optim_state = (np.array(svi_state.optim_state[0]), *(optim_state[1:]))
    svi_state = SVIState(optim_state, svi_state.rng_key)
    ####################################################


    def loop_it(start, stop, fn, init_val, do_jit=True):
        if do_jit:
            return jax.lax.fori_loop(start, stop, fn, init_val)
        else:
            with jax.disable_jit():
                val = init_val
                for i in range(start, stop):
                    val = fn(i, val)
                return val


    @jax.jit
    def train_epoch(num_epoch_iter, svi_state, batchify_state):
        def train_iteration(i, state_and_loss):
            svi_state, loss = state_and_loss
            batch_x, = get_batch(i, batchify_state)
            svi_state, iter_loss = svi.update(svi_state, batch_x)
            return (svi_state, loss + iter_loss / num_epoch_iter)

        return loop_it(0, num_epoch_iter, train_iteration, (svi_state, 0.), do_jit=True)

    rng, epochs_rng = jax.random.split(rng)

    for i in range(num_epochs):
        batchify_rng = jax.random.fold_in(epochs_rng, i)
        num_batches, batchify_state = init_batching(batchify_rng)

        svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
        if np.isnan(loss):
            raise InferenceException
        loss /= data.shape[0]
        print("epoch {}: loss {}".format(i, loss))

    return svi.get_params(svi_state)

def train_model(rng, model, guide, data, batch_size, dp_scale, num_epochs, clipping_threshold=1.):
    """ trains a given model using DPSVI and the globally defined parameters and data """

    optimizer = Adam(1e-3)

    svi = DPSVI(
        model, guide, optimizer, ELBO(),
        num_obs_total=data.shape[0], clipping_threshold=clipping_threshold,
        dp_scale=dp_scale
    )

    return _train_model(rng, svi, data, batch_size, num_epochs)

def train_model_no_dp(rng, model, guide, data, batch_size, num_epochs, **kwargs):
    """ trains a given model using SVI (no DP!) and the globally defined parameters and data """

    optimizer = Adam(1e-3)

    svi = SVI(
        model, guide,
        optimizer, ELBO(),
        num_obs_total=data.shape[0]
    )

    return _train_model(rng, svi, data, batch_size, num_epochs)
