import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro.primitives import sample
from twinify.model_loading import TModelFunction
import argparse
from typing import Iterable

def model_factory(args: argparse.Namespace, orig_data: pd.DataFrame) -> TModelFunction:
    d = orig_data.shape[-1]
    print(f"Privacy parameter epsilon is {args.epsilon}")

    def model(z = None, num_obs_total = None) -> None:
        mu = sample('mu', dists.Normal(args.prior_mu).expand_by((d,)).to_event(1))
        sigma = sample('sigma', dists.InverseGamma(1.).expand_by((d,)).to_event(1))
        sample('x', dists.Normal(mu, sigma), obs=z)

    return model

