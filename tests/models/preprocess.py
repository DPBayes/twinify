import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd

def preprocess(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = ['new_first', 'new_second']
    y['new_first'] += 2
    return y

def model(z = None, num_obs_total = None) -> None:
    batch_size = 1
    if z is not None:
        batch_size = z.shape[0]
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    with plate('batch', num_obs_total, batch_size):
        return sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)
