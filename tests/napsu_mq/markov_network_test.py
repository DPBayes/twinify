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
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from twinify.napsu_mq.marginal_query import *
from maximum_entropy_distribution import MaximumEntropyDistribution
from twinify.napsu_mq.markov_network import MarkovNetwork


class MarkovNetworkTest(unittest.TestCase):
    def setUp(self):
        self.data_gen = BinaryLogisticRegressionDataGenerator(jnp.arange(4).astype(jnp.double))
        self.queries = FullMarginalQuerySet([(0, 3), (1, 3), (2, 3), (4, 3), (0, 1)], self.data_gen.values_by_feature)
        self.full_queries = all_marginals([(0, 1, 2, 3, 4)], self.data_gen.values_by_feature)
        self.mn = MarkovNetwork(self.data_gen.values_by_feature, self.queries)
        self.med = MaximumEntropyDistribution(self.data_gen.values_by_feature, self.queries.flatten())

        self.n_queries = len(self.queries.flatten().queries)
        self.lambda1 = jnp.zeros(self.n_queries)
        self.lambda2 = jnp.ones(self.n_queries)
        self.lambda3 = jnp.arange(self.n_queries).astype(jnp.double)
        self.lambda4 = jnp.arange(self.n_queries).astype(jnp.double) / self.n_queries

    def test_lambda0(self):
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda1)), self.med.lambda0(self.lambda1).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda2)), self.med.lambda0(self.lambda2).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda3)), self.med.lambda0(self.lambda3).item())
        self.assertAlmostEqual(self.mn.lambda0(jnp.array(self.lambda4)), self.med.lambda0(self.lambda4).item())

    def test_suff_stat_mean(self):
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda1)),
                                     jnp.array(self.med.suff_stat_mean(self.lambda1))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda2)),
                                     jnp.array(self.med.suff_stat_mean(self.lambda2))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda3)),
                                     jnp.array(self.med.suff_stat_mean(self.lambda3))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_mean(jnp.array(self.lambda4)),
                                     jnp.array(self.med.suff_stat_mean(self.lambda4))))

    def test_suff_stat_cov(self):
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda1)),
                                     jnp.array(self.med.suff_stat_cov(self.lambda1))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda2)),
                                     jnp.array(self.med.suff_stat_cov(self.lambda2))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda3)),
                                     jnp.array(self.med.suff_stat_cov(self.lambda3))))
        self.assertTrue(jnp.allclose(self.mn.suff_stat_cov(jnp.array(self.lambda4)),
                                     jnp.array(self.med.suff_stat_cov(self.lambda4))))

    def test_suff_stat_mean_and_cov(self):
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda1))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda1))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda1))))
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda2))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda2))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda2))))
        mean, cov = self.mn.suff_stat_mean_and_cov(jnp.array(self.lambda3))
        self.assertTrue(jnp.allclose(mean, self.mn.suff_stat_mean(jnp.array(self.lambda3))))
        self.assertTrue(jnp.allclose(cov, self.mn.suff_stat_cov(jnp.array(self.lambda3))))

    def test_sample(self):
        n = 2000
        sample = jnp.array(self.mn.sample(rng=jax.random.PRNGKey(7438742), lambdas=jnp.array(self.lambda4), n_sample=n))
        sample_one_hot = self.full_queries(sample)
        sample_pmf = sample_one_hot.sum(axis=0) / n
        self.assertTrue(jnp.allclose(sample_pmf, self.med.pmf_all_values(self.lambda4), atol=0.05))


if __name__ == "__main__":
    unittest.main()
