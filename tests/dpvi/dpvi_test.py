import unittest

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.primitives import sample, plate, param
from numpyro import distributions as dists
import pandas as pd
import tempfile

import d3p.random

from twinify.dpvi import DPVIModel, DPVIResult, PrivacyLevel

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
        dpvi_model = DPVIModel(model, output_sample_sites)
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

        self.assertTrue(np.allclose(xs_df.to_numpy().mean(0), merged_df.to_numpy().mean(0), atol=1e-1))
        self.assertTrue(np.allclose(xs_df.to_numpy().std(0, ddof=-1), merged_df.to_numpy().std(0, ddof=-1), atol=1e-1))


class DPVIResultTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

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

        self.model = model

        def guide(xs = None, num_obs_total = None):
            mu_loc = param("mu_loc", jnp.zeros((2,)))
            mu_std = param("mu_std", jnp.zeros((2,)) + .1, constraint=dists.constraints.positive)
            sample('mu', dists.Normal(mu_loc, mu_std))

            sig_loc = param("sig_loc", jnp.zeros((2,)))
            sig_std = param("sig_std", jnp.zeros((2,)) + .1, constraint=dists.constraints.positive)
            sig_unconstrained = sample('sig_unconstrained', dists.Normal(sig_loc, sig_std))
            sig = dists.transforms.biject_to(dists.constraints.positive)(sig_unconstrained)
            sample('sig', dists.Delta(sig))

        self.guide = guide

        self.params = {
            'mu_loc': jnp.array([0.3, -0.2]),
            'mu_std': jnp.array([0.1, 0.1]),
            'sig_loc': jnp.array([-0.12237, 0.235]),
            'sig_std': jnp.array([0.1, 0.1]),
        }

        self.output_sample_sites = ["xs"]
        self.privacy_params = PrivacyLevel(1., 1e-4, 2.1)
        self.final_elbo = 1.67

    def test_init(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.output_sample_sites, self.privacy_params, self.final_elbo
        )

        self.assertTrue(
            jax.tree_util.tree_all(
                jax.tree_util.tree_map(jnp.allclose, self.params, result.parameters)
            )
        )
        self.assertEqual(self.privacy_params, result.privacy_level)
        self.assertEqual(self.final_elbo, result.final_elbo)

    def test_generate(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.output_sample_sites, self.privacy_params, self.final_elbo
        )

        num_data_per_parameter = 100
        num_parameter_samples = 2
        syn_data_sets = list(result.generate_extended(
            d3p.random.PRNGKey(1142), num_data_per_parameter, num_parameter_samples, single_dataframe=False
        ))

        self.assertEqual(2, len(syn_data_sets))
        for syn_data in syn_data_sets:
            self.assertIsInstance(syn_data, pd.DataFrame)
            self.assertEqual(syn_data.shape, (num_data_per_parameter, 2))

    def test_generate_extended_single_dataset(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.output_sample_sites, self.privacy_params, self.final_elbo
        )

        num_data_per_parameter = 100
        num_parameter_samples = 2
        syn_data = result.generate_extended(
            d3p.random.PRNGKey(1142), num_data_per_parameter, num_parameter_samples, single_dataframe=True
        )

        print(type(syn_data))
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertEqual(syn_data.shape, (num_parameter_samples * num_data_per_parameter, 2))


    def test_store_and_load(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.output_sample_sites, self.privacy_params, self.final_elbo
        )

        with tempfile.TemporaryFile("w+b") as f:
            result.store(f)

            f.seek(0)
            loaded_result = DPVIResult.load(f, model=self.model, guide=self.guide)

            self.assertTrue(
                jax.tree_util.tree_all(
                    jax.tree_util.tree_map(jnp.allclose, self.params, loaded_result.parameters)
                )
            )
            self.assertEqual(self.privacy_params, loaded_result.privacy_level)
            self.assertEqual(self.final_elbo, loaded_result.final_elbo)


            result_samples = result.generate(d3p.random.PRNGKey(567), 10, 1)
            loaded_result_samples = loaded_result.generate(d3p.random.PRNGKey(567), 10, 1)
            self.assertTrue(
                np.allclose(result_samples[0].values, loaded_result_samples[0].values)
            )


if __name__ == '__main__':
    unittest.main()
