import unittest
import pandas as pd
import jax
import numpy as np
from twinify.dpvi.sampling import sample_synthetic_data

import numpyro.distributions as dists
from numpyro import sample, plate


def model():
    mu = sample('mu', dists.Normal())
    with plate('batch', 10, 1):
        x = sample('x', dists.Normal(mu, 1).expand_by((1,5)).to_event(1))


def guide():
    sample('mu', dists.Delta(2.))


class SamplingTests(unittest.TestCase):

    def test_sampling(self) -> None:
        samples = sample_synthetic_data(model, guide, {}, jax.random.PRNGKey(0), 2, 3)
        self.assertEqual(set(samples.keys()), {'x'})
        self.assertIsInstance(samples['x'], np.ndarray)
        self.assertEqual(samples['x'].shape, (2, 3, 5))
