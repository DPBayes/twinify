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
from pathlib import Path

import d3p.random
import jax.random
import numpy as np
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile, TemporaryFile
from binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from twinify.napsu_mq.napsu_mq import NapsuMQResult, NapsuMQModel, NapsuMQInferenceConfig
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.dataframe_data import DataDescription


class TestNapsuMQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        jax_rng = jax.random.PRNGKey(22325127)
        cls.data = np.array(data_gen.generate_data(n=2000, rng_key=jax_rng))
        binary_cat_dtype = pd.CategoricalDtype([0, 1], ordered=True)
        cls.dataframe = pd.DataFrame(cls.data, columns=['A', 'B', 'C'])
        for c in cls.dataframe.columns:
            cls.dataframe[c] = cls.dataframe[c].astype(binary_cat_dtype)

        cls.n, cls.d = cls.data.shape

    def setUp(self):
        self.data = self.__class__.data
        self.dataframe = self.__class__.dataframe
        self.n = self.__class__.n
        self.d = self.__class__.d

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_without_IO(self):
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(54363731)
        inference_rng, sampling_rng = d3p.random.split(rng)

        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        datasets = result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_with_IO(self):
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(69700241)
        inference_rng, sampling_rng = d3p.random.split(rng)
        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        napsu_result_file = NamedTemporaryFile("wb")
        with open(napsu_result_file.name, 'wb') as file:
            result.store(file)

        self.assertTrue(Path(napsu_result_file.name).exists())
        self.assertTrue(Path(napsu_result_file.name).is_file())

        napsu_result_read_file = open(napsu_result_file.name, "rb")
        loaded_result: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
        napsu_result_file.close()

        datasets = loaded_result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()

            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

            df_result_file = NamedTemporaryFile()
            df.to_csv(df_result_file)

            self.assertTrue(Path(df_result_file.name).exists())
            self.assertTrue(Path(df_result_file.name).is_file())

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_for_storing_defects(self):
        # Expect model to generate the same results before storing the model and after storing and loading the model
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(74249069)
        inference_rng, sampling_rng = d3p.random.split(rng)
        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        # Use the sampling rng with both generate calls to expect the same generation outcome
        datasets_before_loading = result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets_before_loading), 5)
        self.assertEqual(datasets_before_loading[0].shape, (500, 3))

        napsu_result_file = NamedTemporaryFile("wb")
        with open(napsu_result_file.name, 'wb') as file:
            result.store(file)

        self.assertTrue(Path(napsu_result_file.name).exists())
        self.assertTrue(Path(napsu_result_file.name).is_file())

        napsu_result_read_file = open(napsu_result_file.name, "rb")
        loaded_result: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
        napsu_result_file.close()

        datasets_after_loading = loaded_result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets_after_loading), 5)
        self.assertEqual(datasets_after_loading[0].shape, (500, 3))

        for i, datasets in enumerate(
                list(zip(datasets_before_loading, datasets_after_loading))):
            dataset_before_loading, dataset_after_loading = datasets

            pd.testing.assert_frame_equal(dataset_before_loading, dataset_after_loading)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_with_laplace_plus_mcmc_without_IO(self):
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(85532350)
        inference_rng, sampling_rng = d3p.random.split(rng)

        config = NapsuMQInferenceConfig(method="laplace+mcmc")
        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals, inference_config=config)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        datasets = result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_with_laplace_approximation_without_IO(self):
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(897236)
        inference_rng, sampling_rng = d3p.random.split(rng)

        config = NapsuMQInferenceConfig(method="laplace")
        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals, inference_config=config)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        datasets = result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=False
        )

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_without_IO_single_dataset(self):
        required_marginals = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(85511235)
        inference_rng, sampling_rng = d3p.random.split(rng)

        config = NapsuMQInferenceConfig(method="laplace+mcmc")
        model = NapsuMQModel(forced_queries_in_automatic_selection=required_marginals, inference_config=config)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)))

        dataset = result.generate(
            rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5, single_dataframe=True
        )

        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertEqual(dataset.shape, (2500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        means = dataset.mean()
        stds = dataset.std()
        pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
        pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)

    def test_NAPSUMQ_model_fit_rejects_pure_integer_data(self) -> None:
        # adapted from issue 50
        n = 4

        data = pd.DataFrame({
            'A': np.random.randint(500, 1000, size=n),
            'B': np.random.randint(500, 1000, size=n),
            'C': np.random.randint(500, 1000, size=n)
        }, dtype='category')

        data['A'] = data['A'].astype(int)

        rng = d3p.random.PRNGKey(42)

        model = NapsuMQModel(forced_queries_in_automatic_selection=[])
        with self.assertRaises(ValueError):
            model.fit(data=data, rng=rng, epsilon=1, delta=(n ** (-2)))


class TestNapsuMQResult(unittest.TestCase):

    def test_generate_single_df(self) -> None:
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        domain_sizes = { k: len(cdtype.categories) for k, cdtype in categories.items() }
        domain = { k: np.arange(s) for k, s in domain_sizes.items() }
        data_description = DataDescription(categories)

        posterior_values = np.zeros((1000, 2), dtype=int)
        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain_sizes), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100)

        self.assertIsInstance(samples, pd.DataFrame)
        self.assertEqual(samples.shape, (100, 2))
        self.assertEqual(tuple(samples.columns), ('A', 'B'))
        self.assertEqual(samples['A'].dtype, categories['A'])
        self.assertEqual(samples['B'].dtype, categories['B'])

    def test_generate_multi_df(self) -> None:
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        domain_sizes = { k: len(cdtype.categories) for k, cdtype in categories.items() }
        domain = { k: np.arange(s) for k, s in domain_sizes.items() }

        posterior_values = np.zeros((1000, 2), dtype=int)
        data_description = DataDescription(categories)

        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain_sizes), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100, num_data_per_parameter_sample=20,
                                  single_dataframe=False)

        self.assertEqual(100, len(samples))
        self.assertEqual(samples[0].shape, (20, 2))
        self.assertEqual(tuple(samples[0].columns), ('A', 'B'))

    def test_store_and_load(self) -> None:
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        domain_sizes = { k: len(cdtype.categories) for k, cdtype in categories.items() }
        domain = { k: np.arange(s) for k, s in domain_sizes.items() }

        posterior_values = np.zeros((1000, 2), dtype=int)
        data_description = DataDescription(categories)

        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain_sizes), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100)

        with TemporaryFile('w+b') as f:
            result.store(f)

            f.seek(0)
            loaded_result = NapsuMQResult.load(f)

        loaded_samples = loaded_result.generate(d3p.random.PRNGKey(15412), 100)

        self.assertTrue(np.all(samples.values == loaded_samples.values))


class TestNapsuMQInferenceConfig(unittest.TestCase):

    def test_correct_methods(self):
        config = NapsuMQInferenceConfig()
        config.method = "mcmc"
        config.method = "laplace"
        config.method = "laplace+mcmc"

    def test_incorrect_methods(self):
        config = NapsuMQInferenceConfig()
        with pytest.raises(ValueError):
            config.method = "hfjsdhfk"

    def test_correct_no_laplace_config(self):
        config = NapsuMQInferenceConfig()
        config.method = "mcmc"
        config.laplace_approximation_config = None

    def test_incorrect_no_laplace_config(self):
        config = NapsuMQInferenceConfig()
        config.method = "laplace"
        with pytest.raises(ValueError):
            config.laplace_approximation_config = None

        config = NapsuMQInferenceConfig()
        config.method = "laplace+mcmc"
        with pytest.raises(ValueError):
            config.laplace_approximation_config = None

    def test_correct_no_mcmc_config(self):
        config = NapsuMQInferenceConfig()
        config.method = "laplace"
        config.mcmc_config = None

    def test_incorrect_no_mcmc_config(self):
        config = NapsuMQInferenceConfig()
        config.method = "mcmc"
        with pytest.raises(ValueError):
            config.mcmc_config = None

        config = NapsuMQInferenceConfig()
        config.method = "laplace+mcmc"
        with pytest.raises(ValueError):
            config.mcmc_config = None

    def test_remove_config_then_change_method(self):
        config = NapsuMQInferenceConfig()
        config.method = "mcmc"
        config.laplace_approximation_config = None
        with pytest.raises(ValueError):
            config.method = "laplace"
        with pytest.raises(ValueError):
            config.method = "laplace+mcmc"