import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample
import pandas as pd
from typing import Tuple, Iterable

def preprocess(orig_data: pd.DataFrame) -> Tuple[Iterable[pd.DataFrame], int]:
    return (orig_data['first'], orig_data['second']), len(orig_data)

def postprocess(syn_data: pd.DataFrame) -> pd.DataFrame:
    return syn_data.copy()

def model(x_first = None, x_second = None, num_obs_total = None) -> None:
    mu = sample('mu', dists.Normal())
    sigma = sample('sigma', dists.InverseGamma(1.))
    sample('x_first', dists.Normal(mu, sigma), obs=x_first)
    sample('x_second', dists.Normal(mu, sigma), obs=x_second)
