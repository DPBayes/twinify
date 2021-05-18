import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample
import pandas as pd

def model(z = None, num_obs_total = None) -> None:
    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    sample('x', dists.Normal(mu, sigma), obs=z)
