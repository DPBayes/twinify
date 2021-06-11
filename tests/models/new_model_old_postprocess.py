import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd
from typing import Tuple, Iterable

def preprocess(orig_data: pd.DataFrame) -> Tuple[Iterable[pd.DataFrame], int]:
    return (orig_data['first'].to_frame(), orig_data['second'].to_frame()), len(orig_data)

def postprocess(syn_data: pd.DataFrame) -> pd.DataFrame:
    return syn_data.copy()

def model(x_first = None, x_second = None, num_obs_total = None) -> None:
    batch_size = 1
    if x_first is not None:
        batch_size = x_first.shape[0]
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal())
    sigma = sample('sigma', dists.InverseGamma(1.))
    with plate('batch', num_obs_total, batch_size):
        sample('x_first', dists.Normal(mu, sigma), obs=x_first)
        sample('x_second', dists.Normal(mu, sigma), obs=x_second)
