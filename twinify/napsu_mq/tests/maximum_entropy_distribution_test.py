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
import torch
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
from twinify.napsu_mq.maximum_entropy_distribution import MaximumEntropyDistribution
from twinify.napsu_mq.binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator
from twinify.napsu_mq.marginal_query import MarginalQuery, QueryList, all_marginals, FullMarginalQuerySet


class MaximumEntropyDistributionTest(unittest.TestCase):
    def setUp(self):
        data_gen = BinaryLogisticRegressionDataGenerator(torch.tensor((1.0, 0.0)))
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

    def test_suff_stat_cache(self):
        true_counts = torch.tensor([2, 1, 1, 1, 1, 1, 1])
        true_array = torch.tensor([
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
            (tuple(self.med.suff_stat_array[i, :].int().numpy()), self.med.suff_stat_count_array[i].int().item()) for i
            in range(true_counts.shape[0])}
        for i in range(true_counts.shape[0]):
            self.assertIn((tuple(true_array[i].numpy()), true_counts[i].item()), med_value_count_pairs)

    def test_lambda0(self):
        self.assertAlmostEqual(self.med.lambda0(torch.zeros(4)).item(), 2.07944154167)
        self.assertAlmostEqual(self.med.lambda0(torch.ones(4)).item(), 3.319670555596)

    def test_pmf_all_values_sum(self):
        self.assertAlmostEqual(self.med.pmf_all_values(torch.zeros(4)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med.pmf_all_values(torch.ones(4)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med.pmf_all_values(torch.arange(4).double()).sum().item(), 1.0)

    def test_pmf_all_values(self):
        self.assertTrue(torch.allclose(self.med.pmf_all_values(torch.tensor((1.0, 0.0, 1.0, 0.0))), torch.tensor(
            [0.0560990313, 0.0560990313, 0.0560990313, 0.1524929773, 0.0560990313,
             0.1524929773, 0.0560990313, 0.4145188891])))

    def test_pmf_all_values_sum_canon_query(self):
        self.assertAlmostEqual(self.med_canon.pmf_all_values(torch.zeros(7)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med_canon.pmf_all_values(torch.ones(7)).sum().item(), 1.0)
        self.assertAlmostEqual(self.med_canon.pmf_all_values(torch.arange(7).double()).sum().item(), 1.0)

    def test_suff_stat_mean(self):
        self.assertTrue(torch.allclose(self.med.suff_stat_mean(torch.zeros(4)), torch.tensor((0.25, 0.25, 0.25, 0.25))))
        self.assertTrue(torch.allclose(self.med.suff_stat_mean(torch.ones(4)),
                                       torch.tensor([0.3655292893, 0.3655292893, 0.3655292893, 0.3655292893])))
        self.assertTrue(torch.allclose(self.med.suff_stat_mean(torch.tensor((1.0, 0, 1, 0))),
                                       torch.tensor([0.5670118664, 0.1121980625, 0.5670118664, 0.1121980625])))

    def test_suff_stat_cov(self):
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.zeros(4)),
                                       torch.tensor([[0.1875000000, -0.0625000000, 0.0625000000, -0.0625000000],
                                                     [-0.0625000000, 0.1875000000, -0.0625000000, 0.0625000000],
                                                     [0.0625000000, -0.0625000000, 0.1875000000, -0.0625000000],
                                                     [-0.0625000000, 0.0625000000, -0.0625000000, 0.1875000000]])))
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.ones(4)),
                                       torch.tensor([[0.2319176280, -0.1336116613, 0.1336116613, -0.1336116613],
                                                     [-0.1336116613, 0.2319176280, -0.1336116613, 0.1336116613],
                                                     [0.1336116613, -0.1336116613, 0.2319176280, -0.1336116613],
                                                     [-0.1336116613, 0.1336116613, -0.1336116613, 0.2319176280]])))
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.tensor((1.0, 0.0, 1.0, 0.0))),
                                       torch.tensor([[0.2455094098, -0.0636176328, 0.0930164325, -0.0636176328],
                                                     [-0.0636176328, 0.0996096573, -0.0636176328, 0.0435106260],
                                                     [0.0930164325, -0.0636176328, 0.2455094098, -0.0636176328],
                                                     [-0.0636176328, 0.0435106260, -0.0636176328, 0.0996096573]])))

    def test_suff_stat_mean_and_cov(self):
        lambdas = torch.zeros(4)
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(torch.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(torch.allclose(cov, self.med.suff_stat_cov(lambdas)))

        lambdas = torch.ones(4)
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(torch.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(torch.allclose(cov, self.med.suff_stat_cov(lambdas)))

        lambdas = torch.arange(4).double()
        mean, cov = self.med.suff_stat_mean_and_cov(lambdas)
        self.assertTrue(torch.allclose(mean, self.med.suff_stat_mean(lambdas)))
        self.assertTrue(torch.allclose(cov, self.med.suff_stat_cov(lambdas)))

    def test_conjugate_unnorm_logpdf(self):
        self.assertAlmostEqual(
            self.med.conjugate_unnorm_logpdf(torch.tensor((1.0, 0.0, 1.0, 0.0)), torch.ones(4), 5).item(),
            -12.4031836730)

    def test_sample_inds(self):
        lambdas = torch.tensor((1.0, 0.0, 1.0, 0.0))
        n = 20000
        self.assertTrue(
            torch.allclose(self.med.sample_inds(lambdas, n).bincount() / n, self.med.pmf_all_values(lambdas),
                           atol=0.05))

    def test_sample_inds_canon(self):
        lambdas = torch.tensor((1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0))
        n = 20000
        self.assertTrue(
            torch.allclose(self.med_canon.sample_inds(lambdas, n).bincount() / n,
                           self.med_canon.pmf_all_values(lambdas), atol=0.05)
        )

    def test_mean_query_values(self):
        queries = QueryList([
            MarginalQuery((0, 1, 2), (0, 0, 1)),
            MarginalQuery((0, 1, 2), (1, 0, 0)),
            MarginalQuery((0, 1, 2), (0, 1, 0)),
            MarginalQuery((0, 1, 2), (1, 0, 1)),
        ])
        lambdas = torch.zeros(4)
        samples = self.med.sample(lambdas, n=20000)
        sample_mean = queries(samples).double().mean(dim=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(torch.allclose(sample_mean, computed_mean, atol=0.05))

        lambdas = torch.ones(4)
        samples = self.med.sample(lambdas, n=20000)
        sample_mean = queries(samples).double().mean(dim=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(torch.allclose(sample_mean, computed_mean, atol=0.05))

        lambdas = torch.arange(4).double()
        samples = self.med.sample(lambdas, n=20000)
        sample_mean = queries(samples).double().mean(dim=0)
        computed_mean = self.med.mean_query_values(queries, lambdas)
        self.assertTrue(torch.allclose(sample_mean, computed_mean, atol=0.05))


class MaximumEntropyDistributionWithMultinomialTest(unittest.TestCase):
    def setUp(self):
        data_gen = BinaryLogisticRegressionDataGenerator(torch.tensor((1.0, 0.0)))
        x_values = data_gen.x_values
        queries = QueryList([MarginalQuery((0, 1, 2), x_value) for x_value in x_values])
        queries.queries = queries.queries[0:-1]
        self.med = MaximumEntropyDistribution(data_gen.values_by_feature, queries)

    def mn_mean(self, lambdas):
        return F.softmax(torch.cat((lambdas, torch.zeros(1))), dim=0)

    def mn_cov(self, lambdas):
        d = lambdas.shape[0]
        p = F.softmax(torch.cat((lambdas, torch.zeros(1))), dim=0)
        cov = -torch.outer(p, p)
        cov[torch.eye(d + 1).bool()] = p * (1 - p)
        return cov

    def med_mean_to_mn_mean(self, med_mean):
        return torch.cat([med_mean, torch.tensor((1 - med_mean.sum(),))])

    def test_suff_stat_mean_with_multinomial(self):
        self.assertTrue(torch.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(torch.zeros(7))),
                                       self.mn_mean(torch.zeros(7))))
        self.assertTrue(torch.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(torch.ones(7))),
                                       self.mn_mean(torch.ones(7))))
        self.assertTrue(torch.allclose(self.med_mean_to_mn_mean(self.med.suff_stat_mean(torch.arange(7).double())),
                                       self.mn_mean(torch.arange(7).double())))

    def test_suff_stat_cov_with_multinomial(self):
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.zeros(7)), self.mn_cov(torch.zeros(7))[0:7, 0:7]))
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.ones(7)), self.mn_cov(torch.ones(7))[0:7, 0:7]))
        self.assertTrue(torch.allclose(self.med.suff_stat_cov(torch.arange(7).double()),
                                       self.mn_cov(torch.arange(7).double())[0:7, 0:7]))

    def test_pmf_all_values_with_multinomial(self):
        self.assertTrue(torch.allclose(self.med.pmf_all_values(torch.zeros(7)), self.mn_mean(torch.zeros(7))))
        self.assertTrue(torch.allclose(self.med.pmf_all_values(torch.ones(7)), self.mn_mean(torch.ones(7))))
        self.assertTrue(
            torch.allclose(self.med.pmf_all_values(torch.arange(7).double()), self.mn_mean(torch.arange(7).double())))


if __name__ == "__main__":
    unittest.main()
