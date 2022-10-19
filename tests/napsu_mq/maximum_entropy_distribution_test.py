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

import numpy as np
from scipy.special import softmax
import jax

from maximum_entropy_distribution import MaximumEntropyDistribution
from binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from twinify.napsu_mq.marginal_query import MarginalQuery, QueryList, all_marginals, FullMarginalQuerySet


class MaximumEntropyDistributionTest(unittest.TestCase):
    def setUp(self):
        data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        queries = QueryList([
            MarginalQuery([0, 2], (1, 1)),
            MarginalQuery([0, 2], (1, 0)),
            MarginalQuery([1, 2], (1, 1)),
            MarginalQuery([1, 2], (1, 0)),
        ])
        self.med = MaximumEntropyDistribution(data_gen.values_by_feature, queries)

        queries_canon = FullMarginalQuerySet([(0, 1, 2)], data_gen.values_by_feature)
        queries_canon = queries_canon.get_canonical_queries().flatten()
        self.med_canon = MaximumEntropyDistribution(data_gen.values_by_feature, queries_canon)

        queries_with_full_marginals = all_marginals([(0, 2), (1, 2)], data_gen.values_by_feature)
        self.med_with_full_marginals = MaximumEntropyDistribution(data_gen.values_by_feature,
                                                                  queries_with_full_marginals)

        self.rng_key = jax.random.PRNGKey(7438742)

    def test_suff_stat_cache(self):
        true_counts = np.array([2, 1, 1, 1, 1, 1, 1])
        true_array = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]]
        )
        self.assertEqual(tuple(true_counts.shape), tuple(self.med.suff_stat_count_array.shape))
        self.assertEqual(tuple(true_array.shape), tuple(self.med.suff_stat_array.shape))

        med_value_count_pairs = {
            (tuple(self.med.suff_stat_array[i, :].astype(int)), self.med.suff_stat_count_array[i].astype(int)) for i
            in range(true_counts.shape[0])}
        for i in range(true_counts.shape[0]):
            self.assertIn((tuple(true_array[i]), true_counts[i].item()), med_value_count_pairs)

    def test_lambda0(self):
        self.assertAlmostEqual(self.med.lambda0(np.zeros(4)).item(), 2.07944154167)
        self.assertAlmostEqual(self.med.lambda0(np.ones(4)).item(), 3.319670555596)

    def test_pmf_all_values_sum(self):
        self.assertAlmostEqual(self.med.pmf_all_values(np.zeros(4)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med.pmf_all_values(np.ones(4)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med.pmf_all_values(np.arange(4).astype(np.double)).sum().item(), 1.0)

    def test_pmf_all_values(self):
        self.assertTrue(np.allclose(self.med.pmf_all_values(np.array([1.0, 0.0, 1.0, 0.0])), np.array(
            [0.0560990313, 0.0560990313, 0.0560990313, 0.1524929773, 0.0560990313,
             0.1524929773, 0.0560990313, 0.4145188891])))

    def test_pmf_all_values_sum_canon_query(self):
        self.assertAlmostEqual(self.med_canon.pmf_all_values(np.zeros(7)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med_canon.pmf_all_values(np.ones(7)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med_canon.pmf_all_values(np.arange(7).astype(np.double)).sum().item(), 1.0)

    def test_suff_stat_mean(self):
        self.assertTrue(np.allclose(self.med.suff_stat_mean(np.zeros(4)), np.array([0.25, 0.25, 0.25, 0.25])))
        self.assertTrue(np.allclose(self.med.suff_stat_mean(np.ones(4)),
                                    np.array([0.3655292893, 0.3655292893, 0.3655292893, 0.3655292893])))
        self.assertTrue(np.allclose(self.med.suff_stat_mean(np.array([1.0, 0, 1, 0])),
                                    np.array([0.5670118664, 0.1121980625, 0.5670118664, 0.1121980625])))

    def test_suff_stat_cov(self):
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.zeros(4)),
                                    np.array([[0.1875000000, -0.0625000000, 0.0625000000, -0.0625000000],
                                              [-0.0625000000, 0.1875000000, -0.0625000000, 0.0625000000],
                                              [0.0625000000, -0.0625000000, 0.1875000000, -0.0625000000],
                                              [-0.0625000000, 0.0625000000, -0.0625000000, 0.1875000000]])))
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.ones(4)),
                                    np.array([[0.2319176280, -0.1336116613, 0.1336116613, -0.1336116613],
                                              [-0.1336116613, 0.2319176280, -0.1336116613, 0.1336116613],
                                              [0.1336116613, -0.1336116613, 0.2319176280, -0.1336116613],
                                              [-0.1336116613, 0.1336116613, -0.1336116613, 0.2319176280]])))
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.array([1.0, 0.0, 1.0, 0.0])),
                                    np.array([[0.2455094098, -0.0636176328, 0.0930164325, -0.0636176328],
                                              [-0.0636176328, 0.0996096573, -0.0636176328, 0.0435106260],
                                              [0.0930164325, -0.0636176328, 0.2455094098, -0.0636176328],
                                              [-0.0636176328, 0.0435106260, -0.0636176328, 0.0996096573]])))

    def test_suff_stat_mean_and_cov(self):
        lambdas = np.zeros(4)
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(np.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(np.allclose(cov, self.med.suff_stat_cov(lambdas)))

        lambdas = np.ones(4)
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(np.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(np.allclose(cov, self.med.suff_stat_cov(lambdas)))

        lambdas = np.arange(4).astype(np.double)
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(np.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(np.allclose(cov, self.med.suff_stat_cov(lambdas)))

    def test_conjugate_unnorm_logpdf(self):
        self.assertAlmostEqual(
            self.med.conjugate_unnorm_logpdf(np.array([1.0, 0.0, 1.0, 0.0]), np.ones(4), 5).item(),
            -12.4031836730)

    def test_sample_inds(self):
        lambdas = np.array([1.0, 0.0, 1.0, 0.0])
        n = 20000

        sample_inds = self.med.sample_inds(self.rng_key, lambdas, n)
        bin_count = np.bincount(self.med.sample_inds(self.rng_key, lambdas, n)) / n
        pmf_values = self.med.pmf_all_values(lambdas)

        self.assertTrue(
            np.allclose(bin_count, pmf_values, atol=0.05)
        )

    def test_sample_inds_canon(self):
        lambdas = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0])
        n = 20000

        sample_inds = self.med_canon.sample_inds(self.rng_key, lambdas, n)
        bin_count = np.bincount(self.med_canon.sample_inds(self.rng_key, lambdas, n)) / n
        pmf_values = self.med_canon.pmf_all_values(lambdas)

        self.assertTrue(
            np.allclose(bin_count, pmf_values, atol=0.05)
        )

    def test_mean_query_values(self):
        queries = QueryList([
            MarginalQuery((0, 1, 2), (0, 0, 1)),
            MarginalQuery((0, 1, 2), (1, 0, 0)),
            MarginalQuery((0, 1, 2), (0, 1, 0)),
            MarginalQuery((0, 1, 2), (1, 0, 1)),
        ])
        lambdas = np.zeros(4)
        samples = self.med.sample(self.rng_key, lambdas, n=20000)
        sample_mean = queries(samples).astype(np.double).mean(axis=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(np.allclose(sample_mean, computed_mean, atol=0.05))

        lambdas = np.ones(4)
        samples = self.med.sample(self.rng_key, lambdas, n=20000)
        sample_mean = queries(samples).astype(np.double).mean(axis=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(np.allclose(sample_mean, computed_mean, atol=0.05))

        lambdas = np.arange(4).astype(np.double)
        samples = self.med.sample(self.rng_key, lambdas, n=20000)
        sample_mean = queries(samples).astype(np.double).mean(axis=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(np.allclose(sample_mean, computed_mean, atol=0.05))


class MaximumEntropyDistributionWithMultinomialTest(unittest.TestCase):
    def setUp(self):
        data_gen = BinaryLogisticRegressionDataGenerator(np.array([1.0, 0.0]))
        x_values = data_gen.x_values
        queries = QueryList([MarginalQuery((0, 1, 2), x_value) for x_value in x_values])
        queries.queries = queries.queries[0:-1]
        self.med = MaximumEntropyDistribution(data_gen.values_by_feature, queries)

    def mn_mean(self, lambdas):
        return softmax(np.concatenate((lambdas, np.zeros(1))), axis=0)

    def mn_cov(self, lambdas):
        d = lambdas.shape[0]
        p = softmax(np.concatenate((lambdas, np.zeros(1))), axis=0)
        cov = -np.outer(p, p)
        cov[np.eye(d + 1).astype(bool)] = p * (1 - p)
        return cov

    def med_mean_to_mn_mean(self, med_mean):
        return np.concatenate([med_mean, np.array([1 - med_mean.sum(), ])])

    def test_suff_stat_mean_with_multinomial(self):
        self.assertTrue(np.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(np.zeros(7))),
                                    self.mn_mean(np.zeros(7))))
        self.assertTrue(np.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(np.ones(7))),
                                    self.mn_mean(np.ones(7))))
        self.assertTrue(np.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(np.arange(7).astype(np.double))),
                                    self.mn_mean(np.arange(7).astype(np.double))))

    def test_suff_stat_cov_with_multinomial(self):
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.zeros(7)), self.mn_cov(np.zeros(7))[0:7, 0:7]))
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.ones(7)), self.mn_cov(np.ones(7))[0:7, 0:7]))
        self.assertTrue(np.allclose(self.med.suff_stat_cov(np.arange(7).astype(np.double)),
                                    self.mn_cov(np.arange(7).astype(np.double))[0:7, 0:7]))

    def test_pmf_all_values_with_multinomial(self):
        self.assertTrue(np.allclose(self.med.pmf_all_values(np.zeros(7)), self.mn_mean(np.zeros(7))))
        self.assertTrue(np.allclose(self.med.pmf_all_values(np.ones(7)), self.mn_mean(np.ones(7))))
        self.assertTrue(
            np.allclose(self.med.pmf_all_values(np.arange(7).astype(np.double)),
                        self.mn_mean(np.arange(7).astype(np.double))))


if __name__ == "__main__":
    unittest.main()
