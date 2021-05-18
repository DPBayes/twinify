import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample
import pandas as pd
from typing import Tuple, Dict

def postprocess(samples: Dict[str, np.ndarray], orig_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    syn_data = pd.DataFrame(samples['x'], columns=orig_data.columns)
    encoded_syn_data = syn_data.copy()
    encoded_syn_data['foo'] += 2
    assert False
    return syn_data, encoded_syn_data

def model(z = None, num_obs_total = None) -> None:
    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    sample('x', dists.Normal(mu, sigma), obs=z)