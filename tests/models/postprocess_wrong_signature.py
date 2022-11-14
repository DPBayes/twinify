import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd
from twinify import DataDescription
from typing import Tuple, Dict, Any

def postprocess(samples: np.ndarray, data_description: DataDescription, and_another: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    syn_data = pd.DataFrame(samples, columns=data_description.columns)
    encoded_syn_data = syn_data.copy()
    encoded_syn_data += 2
    return syn_data, encoded_syn_data

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
