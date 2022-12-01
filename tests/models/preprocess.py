import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd
from typing import Tuple, Iterable

def preprocess(x: pd.DataFrame) -> Tuple[Iterable[pd.DataFrame], int]:
    y = x.copy()
    y.columns = ['new_first', 'new_second']
    y2 = y['new_first'].copy()
    y2.name = 'y2'
    y['new_first'] += 2
    assert(isinstance(y2, pd.Series))
    assert(isinstance(y, pd.DataFrame))
    return (y, y2), len(y)

def model(z = None, z2 = None, num_obs_total = None) -> None:
    batch_size = 1
    if z is not None:
        batch_size = z.shape[0]
        assert(z.shape is not None)
        assert(z.shape[0] == z2.shape[0])
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    with plate('batch', num_obs_total, batch_size):
        sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)