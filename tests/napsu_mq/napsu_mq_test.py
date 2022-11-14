# Copyright 2022 twinify Developers and their Assignees

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
from twinify.napsu_mq.napsu_mq import NapsuMQResult, NapsuMQModel
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.dataframe_data import DataDescription


class TestNapsuMQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        jax_rng = jax.random.PRNGKey(22325127)
        cls.data = data_gen.generate_data(n=2000, rng_key=jax_rng)
        cls.dataframe = pd.DataFrame(cls.data, columns=['A', 'B', 'C'], dtype=int)
        cls.n, cls.d = cls.data.shape

    def setUp(self):
        self.data = self.__class__.data
        self.dataframe = self.__class__.dataframe
        self.n = self.__class__.n
        self.d = self.__class__.d

    # Takes about ~ 1 minute to run
    @pytest.mark.slow
    def test_NAPSUMQ_model_without_IO(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(54363731)
        inference_rng, sampling_rng = d3p.random.split(rng)

        model = NapsuMQModel(column_feature_set=column_feature_set, use_laplace_approximation=False)
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
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(69700241)
        inference_rng, sampling_rng = d3p.random.split(rng, use_laplace_approximation=False)
        model = NapsuMQModel(column_feature_set=column_feature_set)
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
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(74249069)
        inference_rng, sampling_rng = d3p.random.split(rng)
        model = NapsuMQModel(column_feature_set=column_feature_set, use_laplace_approximation=False)
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
    def test_NAPSUMQ_model_with_laplace_approximation_without_IO(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(85532350)
        inference_rng, sampling_rng = d3p.random.split(rng)

        model = NapsuMQModel(column_feature_set=column_feature_set, use_laplace_approximation=True)
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
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(85511235)
        inference_rng, sampling_rng = d3p.random.split(rng)

        model = NapsuMQModel(column_feature_set=column_feature_set, use_laplace_approximation=True)
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


class TestNapsuMQResult(unittest.TestCase):

    def test_generate_single_df(self) -> None:
        domain = {'A': np.arange(4), 'B': np.arange(3)}
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        data_description = DataDescription(categories)

        posterior_values = np.zeros((1000, 2), dtype=int)
        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100)

        self.assertIsInstance(samples, pd.DataFrame)
        self.assertEqual(samples.shape, (100, 2))
        self.assertEqual(tuple(samples.columns), ('A', 'B'))
        self.assertEqual(samples['A'].dtype, categories['A'])
        self.assertEqual(samples['B'].dtype, categories['B'])

    def test_generate_multi_df(self) -> None:
        domain = {'A': np.arange(4), 'B': np.arange(3)}

        posterior_values = np.zeros((1000, 2), dtype=int)
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        data_description = DataDescription(categories)

        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100, num_data_per_parameter_sample=20,
                                  single_dataframe=False)

        self.assertEqual(100, len(samples))
        self.assertEqual(samples[0].shape, (20, 2))
        self.assertEqual(tuple(samples[0].columns), ('A', 'B'))

    def test_store_and_load(self) -> None:
        domain = {'A': np.arange(4), 'B': np.arange(3)}

        posterior_values = np.zeros((1000, 2), dtype=int)
        categories = {'A': pd.CategoricalDtype(('A', 'B', 'C', 'D')), 'B': pd.CategoricalDtype(('x', 'y', 'z'))}
        data_description = DataDescription(categories)

        result = NapsuMQResult(domain, FullMarginalQuerySet([('A', 'B')], domain), posterior_values, data_description)

        samples = result.generate(d3p.random.PRNGKey(15412), 100)

        with TemporaryFile('w+b') as f:
            result.store(f)

            f.seek(0)
            loaded_result = NapsuMQResult.load(f)

        loaded_samples = loaded_result.generate(d3p.random.PRNGKey(15412), 100)

        self.assertTrue(np.all(samples.values == loaded_samples.values))
