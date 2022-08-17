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

import jax.random
import numpy as np
from twinify.napsu_mq.binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
import random


class BinaryLogisticRegressionDataGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.data_generator3d = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        self.data_generator4d = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0, 2.0]))

    def test_values_by_feature(self):
        self.assertDictEqual(self.data_generator3d.values_by_feature, {0: [0, 1], 1: [0, 1], 2: [0, 1]})
        self.assertDictEqual(self.data_generator4d.values_by_feature, {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]})

    def test_x_values_3d(self):
        x_values_tuples = {tuple(x_value) for x_value in self.data_generator3d.x_values}
        self.assertEqual(len(x_values_tuples), 8)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.assertIn((i, j, k), x_values_tuples)

    def test_x_values_4d(self):
        x_values_tuples = {tuple(x_value) for x_value in self.data_generator4d.x_values}
        self.assertEqual(len(x_values_tuples), 16)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.assertIn((i, j, k, l), x_values_tuples)

    def test_data_generation_3d_jax(self,):
        rng_key = jax.random.PRNGKey(random.randint(0, 10000000))

        data = self.data_generator3d.generate_data(100000, rng_key=rng_key)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        self.assertAlmostEqual(0.5, data_mean[0], places=1)
        self.assertAlmostEqual(0.5, data_mean[1], places=1)
        self.assertAlmostEqual(0.615, data_mean[2], places=2)

        self.assertAlmostEqual(0.5, data_std[0], places=2)
        self.assertAlmostEqual(0.5, data_std[1], places=2)
        self.assertAlmostEqual(0.486, data_std[2], places=2)

    def test_data_generation_4d_jax(self):
        rng_key = jax.random.PRNGKey(random.randint(0, 10000000))

        data = self.data_generator4d.generate_data(100000, rng_key=rng_key)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        self.assertAlmostEqual(0.5, data_mean[0], places=1)
        self.assertAlmostEqual(0.5, data_mean[1], places=1)
        self.assertAlmostEqual(0.5, data_mean[2], places=1)
        self.assertAlmostEqual(0.766, data_mean[3], places=2)

        self.assertAlmostEqual(0.5, data_std[0], places=1)
        self.assertAlmostEqual(0.5, data_std[1], places=1)
        self.assertAlmostEqual(0.5, data_std[2], places=1)
        self.assertAlmostEqual(0.423, data_std[3], places=2)


if __name__ == "__main__":
    unittest.main()
