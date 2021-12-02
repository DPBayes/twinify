import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro.distributions as dists
from numpyro.primitives import sample, plate
from twinify.model_loading import TModelFunction
import argparse
from typing import Iterable

def model_factory(twinify_args: argparse.Namespace, unparsed_args: Iterable[str], orig_data: pd.DataFrame) -> None:
    model_args_parser = argparse.ArgumentParser()
    model_args_parser.add_argument('--prior_mu', type=float, default=0.)
    args = model_args_parser.parse_args(unparsed_args, twinify_args)

    d = orig_data.shape[-1]
    print(f"Model using prior mu = {args.prior_mu}")
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
            sample('x', dists.Normal(mu, sigma).to_event(1), obs=z)

    return (model, 7)
