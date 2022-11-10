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
from tempfile import NamedTemporaryFile
from binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from twinify.napsu_mq.napsu_mq import NapsuMQResult, NapsuMQModel


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

        model = NapsuMQModel(column_feature_set=column_feature_set)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)),
                           use_laplace_approximation=False)


        datasets = result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5)

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
        inference_rng, sampling_rng = d3p.random.split(rng)
        model = NapsuMQModel(column_feature_set=column_feature_set)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)),
                           column_feature_set=column_feature_set,
                           use_laplace_approximation=False)

        napsu_result_file = NamedTemporaryFile("wb")
        with open(napsu_result_file.name, 'wb') as file:
            result.store(file)

            self.assertTrue(Path(napsu_result_file.name).exists())
            self.assertTrue(Path(napsu_result_file.name).is_file())

        napsu_result_read_file = open(napsu_result_file.name, "rb")
        loaded_result: NapsuMQResult = NapsuMQResult.load(napsu_result_read_file)
        napsu_result_file.close()

        datasets = loaded_result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5)

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
    def test_NAPSUMQ_model_with_laplace_approximation_without_IO(self):
        column_feature_set = [
            ('A', 'B'), ('B', 'C'), ('A', 'C')
        ]

        rng = d3p.random.PRNGKey(85532350)
        inference_rng, sampling_rng = d3p.random.split(rng)

        model = NapsuMQModel(column_feature_set=column_feature_set)
        result = model.fit(data=self.dataframe, rng=inference_rng, epsilon=1, delta=(self.n ** (-2)),
                           column_feature_set=column_feature_set,
                           use_laplace_approximation=True)

        datasets = result.generate_extended(rng=sampling_rng, num_data_per_parameter_sample=500, num_parameter_samples=5)

        self.assertEqual(len(datasets), 5)
        self.assertEqual(datasets[0].shape, (500, 3))

        original_means = self.dataframe.mean()
        original_stds = self.dataframe.std()

        for i, df in enumerate(datasets):
            means = df.mean()
            stds = df.std()
            pd.testing.assert_series_equal(means, original_means, check_exact=False, rtol=0.3)
            pd.testing.assert_series_equal(stds, original_stds, check_exact=False, rtol=0.3)
