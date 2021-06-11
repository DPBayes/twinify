import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd
from typing import Tuple, Dict

def postprocess(syn_data: pd.DataFrame) -> None:
    encoded_syn_data = syn_data.copy()
    encoded_syn_data += 2

def model(z = None, num_obs_total = None) -> None:
    batch_size = 1
    if z is not None:
        batch_size = z.shape[0]
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    with plate('batch', num_obs_total, batch_size):
        sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)
