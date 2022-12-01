# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.primitives import sample, plate, param
from numpyro import distributions as dists
import pandas as pd
import tempfile
import pytest

import d3p.random

from twinify.dpvi import DPVIModel, DPVIResult, PrivacyLevel, InferenceException
from twinify.dpvi.modelling import slice_feature
from twinify.dataframe_data import DataDescription


def model(data = None, num_obs_total = None):
    batch_size = 1
    if data is not None:
        batch_size = np.shape(data)[0]
    if num_obs_total is None:
        num_obs_total = batch_size

    mu = sample('mu', dists.Normal(0, 10), sample_shape=(3,))
    sig = sample('sig', dists.InverseGamma(1.), sample_shape=(3,))
    with plate('obs', num_obs_total, batch_size):
        # simulating multiple sample sites that are not sampled in the order they appear in the data
        ys = sample('ys', dists.Normal(mu[0], sig[0]), obs=slice_feature(data, -2, -1)).reshape(-1, 1)
        xs = sample('xs', dists.MultivariateNormal(mu[1:], jnp.diag(sig[1:])), obs=slice_feature(data, 0, -2))
        cats = sample('cats', dists.Categorical(probs=np.array([.5, .5, .5])), obs=slice_feature(data, -1, dtype=jnp.int64)).reshape(-1, 1)

    return jnp.hstack((xs, ys, cats))


class DPVITests(unittest.TestCase):

    def setUp(self) -> None:
        self.data_description = DataDescription({
            'first': np.dtype(np.float64),
            'second': np.dtype(np.float64),
            'third': np.dtype(np.float32),
            'cats!': pd.CategoricalDtype(["cat1", "cat2", "cat3"])
        })

        np.random.seed(82634593)
        L = np.array([[1., 0, 0], [.87, .3, 0], [0, 0, .5]])
        mu = np.array([2., -3., 0])
        xs = np.random.randn(10000, 3) @ L.T + mu
        cs = np.random.choice(3, size=(10000,))
        cs = pd.Series(pd.Categorical.from_codes(cs, dtype=self.data_description.dtypes['cats!']))
        xs_df = pd.DataFrame(xs, columns=('first', 'second', 'third'))
        xs_df['cats!'] = cs
        xs_df['third'] = xs_df['third'].astype(np.float32)
        self.xs_df = xs_df

    @pytest.mark.slow
    def test_inference_and_sampling(self) -> None:
        xs_df = self.xs_df

        epsilon = 4.
        delta = 1e-6

        rng = d3p.random.PRNGKey(96392153)
        dpvi_model = DPVIModel(model, clipping_threshold=10., num_epochs=300, subsample_ratio=0.01)
        dpvi_fit = dpvi_model.fit(xs_df, rng, epsilon, delta, silent=True)

        self.assertEqual(epsilon, dpvi_fit.privacy_level.epsilon)
        self.assertEqual(delta, dpvi_fit.privacy_level.delta)
        self.assertIsNotNone(dpvi_fit.parameters)
        self.assertEqual(self.data_description, dpvi_fit.data_description)

        num_synthetic_data_sets = 10
        num_samples_per_set = 1000
        dfs = dpvi_fit.generate(
            d3p.random.PRNGKey(8902635), num_synthetic_data_sets, num_samples_per_set, single_dataframe=False
        )

        for i, df in enumerate(dfs):
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual((num_samples_per_set, xs_df.shape[1]), df.shape)
            self.assertEqual(tuple(xs_df.columns), tuple(df.columns))
            self.assertTrue((xs_df.dtypes == df.dtypes).all())

        self.assertEqual(num_synthetic_data_sets - 1, i)

        merged_df: pd.DataFrame = pd.concat(dfs)

        self.assertTrue(np.allclose(xs_df[xs_df.columns[:-1]].to_numpy().mean(0), merged_df[merged_df.columns[:-1]].to_numpy().mean(0), atol=1e-1))
        self.assertTrue(np.allclose(xs_df[xs_df.columns[:-1]].to_numpy().std(0, ddof=-1), merged_df[merged_df.columns[:-1]].to_numpy().std(0, ddof=-1), atol=1e-1))

    def test_fit_aborts_for_nan(self) -> None:
        np.random.seed(82634593)
        L = np.array([[1., 0, 0], [.87, .3, 0], [0, 0, .5]])
        mu = np.array([2., -3., 0])
        xs = np.random.randn(100, 3) @ L.T + mu
        xs[0] = np.zeros((xs.shape[-1],)) * np.nan
        xs_df = pd.DataFrame(xs)

        epsilon = 4.
        delta = 1e-6

        rng = d3p.random.PRNGKey(96392153)
        dpvi_model = DPVIModel(model, clipping_threshold=10., num_epochs=1, subsample_ratio=0.1)
        with self.assertRaises(InferenceException):
            dpvi_model.fit(xs_df, rng, epsilon, delta, silent=True)

    def test_fit_works(self) -> None:
        xs_df = self.xs_df
        epsilon = 4.
        delta = 1e-6

        rng = d3p.random.PRNGKey(96392153)
        dpvi_model = DPVIModel(model, clipping_threshold=10., num_epochs=1, subsample_ratio=0.1)
        dpvi_fit = dpvi_model.fit(xs_df, rng, epsilon, delta, silent=False)

        self.assertEqual(epsilon, dpvi_fit.privacy_level.epsilon)
        self.assertEqual(delta, dpvi_fit.privacy_level.delta)
        self.assertTrue(dpvi_fit.privacy_level.dp_noise > 0)
        self.assertIsNotNone(dpvi_fit.parameters)
        self.assertEqual(self.data_description, dpvi_fit.data_description)

        self.assertIn('auto_loc', dpvi_fit.parameters)
        self.assertEqual((6,), dpvi_fit.parameters['auto_loc'].shape)
        self.assertIn('auto_scale', dpvi_fit.parameters)
        self.assertEqual((6,), dpvi_fit.parameters['auto_scale'].shape)


    def test_fit_works_silent(self) -> None:
        xs_df = self.xs_df
        epsilon = 4.
        delta = 1e-6

        rng = d3p.random.PRNGKey(96392153)
        dpvi_model = DPVIModel(model, clipping_threshold=10., num_epochs=1, subsample_ratio=0.1)
        dpvi_fit = dpvi_model.fit(xs_df, rng, epsilon, delta, silent=True)

        self.assertEqual(epsilon, dpvi_fit.privacy_level.epsilon)
        self.assertEqual(delta, dpvi_fit.privacy_level.delta)
        self.assertTrue(dpvi_fit.privacy_level.dp_noise > 0)
        self.assertIsNotNone(dpvi_fit.parameters)
        self.assertEqual(self.data_description, dpvi_fit.data_description)

        self.assertIn('auto_loc', dpvi_fit.parameters)
        self.assertEqual((6,), dpvi_fit.parameters['auto_loc'].shape)
        self.assertIn('auto_scale', dpvi_fit.parameters)
        self.assertEqual((6,), dpvi_fit.parameters['auto_scale'].shape)

    def test_num_iterations_for_epochs(self) -> None:
        num_epochs = 10
        subsample_ratio = 0.33
        expected_num_iter = 30

        num_iter = DPVIModel.num_iterations_for_epochs(num_epochs, subsample_ratio)
        self.assertEqual(expected_num_iter, num_iter)

    def test_num_epochs_for_iterations(self) -> None:
        num_iter = 30
        subsample_ratio = 0.33
        expected_num_epochs = 10

        num_epochs = DPVIModel.num_epochs_for_iterations(num_iter, subsample_ratio)
        self.assertEqual(expected_num_epochs, num_epochs)

    def test_batch_size_for_subsample_ratio(self) -> None:
        num_data = 1000
        subsample_ratio = 0.0301
        expected_batch_size = 30

        batch_size = DPVIModel.batch_size_for_subsample_ratio(subsample_ratio, num_data)
        self.assertEqual(expected_batch_size, batch_size)

    def test_subsample_ratio_for_batch_size(self) -> None:
        num_data = 1000
        batch_size = 30
        expected_subsample_ratio = 0.03

        subsample_ratio = DPVIModel.subsample_ratio_for_batch_size(batch_size, num_data)
        self.assertEqual(expected_subsample_ratio, subsample_ratio)


class DPVIResultTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.model = model

        def guide(data = None, num_obs_total = None):
            mu_loc = param("mu_loc", jnp.zeros((3,)))
            mu_std = param("mu_std", jnp.zeros((3,)) + .1, constraint=dists.constraints.positive)
            sample('mu', dists.Normal(mu_loc, mu_std))

            sig_loc = param("sig_loc", jnp.zeros((3,)))
            sig_std = param("sig_std", jnp.zeros((3,)) + .1, constraint=dists.constraints.positive)
            sig_unconstrained = sample('sig_unconstrained', dists.Normal(sig_loc, sig_std))
            sig = dists.transforms.biject_to(dists.constraints.positive)(sig_unconstrained)
            sample('sig', dists.Delta(sig))

        self.guide = guide

        self.params = {
            'mu_loc': jnp.array([0.3, -0.2, 0.0]),
            'mu_std': jnp.array([0.1, 0.1, 0.1]),
            'sig_loc': jnp.array([-0.12237, 0.235, .44]),
            'sig_std': jnp.array([0.1, 0.1, 0.1]),
        }

        self.privacy_params = PrivacyLevel(1., 1e-4, 2.1)
        self.final_elbo = 1.67

        self.data_description = DataDescription({
            'first': np.dtype(np.float64),
            'second': np.dtype(np.float64),
            'third': np.dtype(np.float32),
            'cats!': pd.CategoricalDtype(["cat1", "cat2", "cat3"])
        })

    def test_init(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.privacy_params, self.final_elbo, self.data_description
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
            self.model, self.guide, self.params, self.privacy_params, self.final_elbo, self.data_description
        )

        num_data_per_parameter = 100
        num_parameter_samples = 2
        syn_data_sets = list(result.generate(
            d3p.random.PRNGKey(1142), num_parameter_samples, num_data_per_parameter, single_dataframe=False
        ))

        self.assertEqual(2, len(syn_data_sets))
        for syn_data in syn_data_sets:
            self.assertIsInstance(syn_data, pd.DataFrame)
            self.assertEqual(syn_data.shape, (num_data_per_parameter, 4))
            self.assertEqual(tuple(self.data_description.columns), tuple(syn_data.columns))
            for col in syn_data.columns:
                self.assertEqual(self.data_description.dtypes[col], syn_data[col].dtype)

    def test_generate_single_dataset(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.privacy_params, self.final_elbo, self.data_description
        )

        num_data_per_parameter = 100
        num_parameter_samples = 2
        syn_data = result.generate(
            d3p.random.PRNGKey(1142), num_parameter_samples, num_data_per_parameter
        )

        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertEqual(syn_data.shape, (num_parameter_samples * num_data_per_parameter, 4))
        self.assertEqual(tuple(self.data_description.columns), tuple(syn_data.columns))
        for col in syn_data.columns:
            self.assertEqual(self.data_description.dtypes[col], syn_data[col].dtype)

    def test_store_and_load(self) -> None:
        result = DPVIResult(
            self.model, self.guide, self.params, self.privacy_params, self.final_elbo, self.data_description
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
            self.assertEqual(self.data_description, loaded_result.data_description)

            result_samples = result.generate(d3p.random.PRNGKey(567), 10, 1)
            loaded_result_samples = loaded_result.generate(d3p.random.PRNGKey(567), 10, 1)
            self.assertTrue(
                (result_samples.values == loaded_result_samples.values).all().all()
            )


if __name__ == '__main__':
    unittest.main()
