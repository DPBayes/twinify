import unittest

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.primitives import sample, plate
from numpyro import distributions as dists
import pandas as pd

import d3p.random

import twinify.dpvi.dpvi_model

class DPVITests(unittest.TestCase):

    def test(self):
        np.random.seed(82634593)
        L = np.array([[1., .87], [0, .3]])
        mu = np.array([2., -3.])
        xs = np.random.randn(10000, 2) @ L.T + mu
        # xs_df = pd.DataFrame(xs, columns=('first', 'second')) # TODO: DPVIResult must do postprocessing for column names; not yet implemented
        xs_df = pd.DataFrame(xs)

        def model(xs = None, num_obs_total = None):
            batch_size = 1
            if xs is not None:
                batch_size = np.shape(xs)[0]
            if num_obs_total is None:
                num_obs_total = batch_size

            mu = sample('mu', dists.Normal(0, 10), sample_shape=(2,))
            sig = sample('sig', dists.InverseGamma(1.), sample_shape=(2,))
            with plate('obs', num_obs_total, batch_size):
                sample('xs', dists.MultivariateNormal(mu, jnp.diag(sig)), obs=xs)

        output_sample_sites = ['xs']

        rng = d3p.random.PRNGKey(96392153)
        dpvi_model = twinify.dpvi.dpvi_model.DPVIModel(model, output_sample_sites)
        dpvi_fit = dpvi_model.fit(xs_df, rng, epsilon = 1., delta = 1e-6, clipping_threshold = 4., num_iter = 30000, q = 0.01)

        num_synthetic_data_sets = 10
        num_samples_per_set = 1000
        dfs = list(dpvi_fit.generate_extended(d3p.random.PRNGKey(8902635), num_samples_per_set, num_synthetic_data_sets))

        for i, df in enumerate(dfs):
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual((num_samples_per_set, xs_df.shape[1]), df.shape)
            self.assertEqual(list(xs_df.columns), list(df.columns)) # TODO: DPVIResult must do postprocessing for column names; not yet implemented

        self.assertEqual(num_synthetic_data_sets - 1, i)

        merged_df: pd.DataFrame = pd.concat(dfs)

        self.assertTrue(np.allclose(xs_df.to_numpy().mean(0), merged_df.to_numpy().mean(0), atol=1e-2))
        self.assertTrue(np.allclose(xs_df.to_numpy().std(0, ddof=-1), merged_df.to_numpy().std(0, ddof=-1), atol=1e-1))


if __name__ == '__main__':
    unittest.main()
