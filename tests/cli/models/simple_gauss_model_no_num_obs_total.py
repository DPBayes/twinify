# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
import pandas as pd

def model(z = None) -> None:
    batch_size = 1
    if z is not None:
        batch_size = z.shape[0]

    mu = sample('mu', dists.Normal().expand_by((2,)).to_event(1))
    sigma = sample('sigma', dists.InverseGamma(1.).expand_by((2,)).to_event(1))
    with plate('batch', batch_size, batch_size):
        return sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)
