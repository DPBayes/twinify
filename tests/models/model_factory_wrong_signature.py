import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
from twinify.dpvi import ModelFunction
from twinify import DataDescription
import argparse
from typing import Iterable

def model_factory(args: argparse.Namespace, data_description: DataDescription) -> ModelFunction:
    d = data_description.num_columns
    print(f"Privacy parameter epsilon is {args.epsilon}")

    def model(z = None, num_obs_total = None) -> None:
        batch_size = 1
        if z is not None:
            batch_size = z.shape[0]
        if num_obs_total is None:
            num_obs_total = batch_size

        mu = sample('mu', dists.Normal(args.prior_mu).expand_by((d,)).to_event(1))
        sigma = sample('sigma', dists.InverseGamma(1.).expand_by((d,)).to_event(1))
        with plate('batch', num_obs_total, batch_size):
            return sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)

    return model

