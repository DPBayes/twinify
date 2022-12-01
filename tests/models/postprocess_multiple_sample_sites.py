import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd
from typing import Tuple, Dict

def postprocess(samples: Dict[str, np.ndarray], orig_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert(samples['x'].ndim == 2)
    assert(samples['y'].ndim == 1)

    syn_data = pd.DataFrame(samples['x'], columns=orig_data.columns)
    syn_data['foo'] = samples['y']
    encoded_syn_data = syn_data.copy()

    encoded_syn_data += 2
    return syn_data, encoded_syn_data

def model(z = None, z2 = None, num_obs_total = None) -> None:
    batch_size = 1
    if z is not None:
        batch_size = z.shape[0]
        assert(z2 is not None)
        assert(z2.shape[0] == batch_size)
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    with plate('batch', num_obs_total, batch_size):
        sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)
        sample('y', dists.Normal().expand_by((1,)).to_event(1), obs=z2)