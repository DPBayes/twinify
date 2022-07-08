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
from functools import reduce
from operator import mul
import torch

torch.set_default_dtype(torch.float64)
from twinify.napsu_mq.marginal_query import *
from twinify.napsu_mq.binary_logistic_regression_generator import BinaryLogisticRegressionDataGenerator


class MarginalQueryTest(unittest.TestCase):
    def setUp(self):
        self.datagen = BinaryLogisticRegressionDataGenerator(torch.tensor((1.0, 2.0)))
        self.data = self.datagen.generate_data(100)
        self.x_values = self.datagen.x_values
        self.values_by_feature = {i: [0, 1] for i in range(3)}
        self.queries = QueryList([
            MarginalQuery([0, 2], (1, 1)),
            MarginalQuery([0, 2], (1, 0)),
            MarginalQuery([1, 2], (1, 1)),
            MarginalQuery([0, 1], (1, 1)),
            MarginalQuery([0, 1], (1, 0)),
        ])

    def test_evaluate_subset(self):
        subset = {0, 2, 3}
        subset_values = self.queries.evaluate_subset(self.data, subset)
        all_values = self.queries(self.data)
        self.assertTrue((subset_values == all_values[:, list(subset)]).all())

    def test_get_subset(self):
        subset = {0, 2, 3}
        subset_values = self.queries.get_subset(subset)(self.data)
        all_values = self.queries(self.data)
        self.assertTrue((subset_values == all_values[:, list(subset)]).all())

    def test_all_marginals_for_feature_set(self):
        marginals = all_marginals_for_feature_set((0, 2), self.values_by_feature)
        marginals_as_tuples = {(tuple(query.inds), tuple(query.value.numpy())) for query in marginals}
        self.assertEqual(len(marginals), 4)
        self.assertEqual(len(marginals_as_tuples), 4)
        for marginal_tuple in marginals_as_tuples:
            self.assertEqual(marginal_tuple[0], (0, 2))
        for i in range(2):
            for j in range(2):
                self.assertIn(((0, 2), (i, j)), marginals_as_tuples)

    def test_all_marginals(self):
        feature_sets = [(0, 1), (0, 2)]
        marginals = all_marginals(feature_sets, self.values_by_feature).queries
        marginals_as_tuples = {(tuple(query.inds), tuple(query.value.numpy())) for query in marginals}
        self.assertEqual(len(marginals), 4 * 2)
        self.assertEqual(len(marginals_as_tuples), 4 * 2)
        for feature_set in feature_sets:
            for i in range(2):
                for j in range(2):
                    self.assertIn((feature_set, (i, j)), marginals_as_tuples)

    def test_get_variable_associations(self):
        correct_result = {(0, 2): {0, 1}, (1, 2): {2}, (0, 1): {3, 4}}
        result = self.queries.get_variable_associations()
        result = {key: set(val) for key, val in result.items()}
        self.assertDictEqual(result, correct_result)

    def test_as_tuple(self):
        for query in self.queries.queries:
            t_ind, t_val = query.as_tuple()
            self.assertTupleEqual(t_ind, tuple(query.inds))
            self.assertEqual(len(t_val), query.value.shape[0])
            self.assertTrue(torch.allclose(torch.tensor(t_val), query.value))


class Domain:
    def __init__(self, values_by_col):
        self.values_by_col = values_by_col
        self.d = len(values_by_col.keys())
        self.size = reduce(mul, [len(col_values) for col_values in self.values_by_col.values()])

    def get_x_values(self):
        x_values = torch.zeros((self.size, self.d))
        for i, val in enumerate(itertools.product(*self.values_by_col.values())):
            x_values[i, :] = torch.tensor(val)
        return x_values


class CanonicalQueriesTest(unittest.TestCase):

    def query_matrix_rank(self, domain, queries):
        mat = queries(domain.get_x_values()).double()
        mat = torch.cat([mat, torch.ones((mat.shape[0], 1))], dim=1)
        return torch.linalg.matrix_rank(mat).item()

    def test_canonical_queries_rank_binary_domain(self):
        binary_domain = Domain({0: range(2), 1: range(2), 2: range(2)})
        full_binary_queries = FullMarginalQuerySet([(0, 1, 2)], binary_domain.values_by_col)
        two_way_marginal_queries = FullMarginalQuerySet([(0, 2), (1, 2)], binary_domain.values_by_col)

        canon_queries = full_binary_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(binary_domain, canon_queries)
        self.assertEqual(n_canon_queries, 7)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = two_way_marginal_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(binary_domain, canon_queries)
        self.assertEqual(n_canon_queries, 5)
        self.assertEqual(rank, n_canon_queries + 1)

    def test_canonical_queries_nonbinary_domain(self):
        domain = Domain({"0": range(2), "1": range(2), "2": range(3), "3": range(4)})
        full_queries = FullMarginalQuerySet([("0", "1", "2", "3")], domain.values_by_col)
        naive_bayes_queries = FullMarginalQuerySet([("0", "1"), ("0", "2"), ("0", "3")], domain.values_by_col)
        naive_bayes_cross_queries = FullMarginalQuerySet([("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("2", "3")],
                                                         domain.values_by_col)

        canon_queries = full_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(n_canon_queries, 2 * 2 * 3 * 4 - 1)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = naive_bayes_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = naive_bayes_cross_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(rank, n_canon_queries + 1)

    def test_canonical_queries_large_nonbinary_domain(self):
        domain = Domain({0: range(2), 1: range(2), 2: range(3), 3: range(4), 4: range(5), 5: range(6)})
        naive_bayes_cross_queries = FullMarginalQuerySet([(0, 1), (0, 2), (0, 3), (1, 4), (2, 5)], domain.values_by_col)
        naive_bayes_3_way_cross_queries = FullMarginalQuerySet([(0, 1), (0, 2), (0, 3), (1, 4, 5), (2, 5)],
                                                               domain.values_by_col)
        naive_bayes_3_way_cross_queries_missing = FullMarginalQuerySet([(0, 1), (0, 2), (0, 3), (1, 4, 3)],
                                                                       domain.values_by_col)
        one_way_marginals = FullMarginalQuerySet([(0,), (1,), (2,), (3,), (4,), (5,)], domain.values_by_col)
        one_way_marginals_missing = FullMarginalQuerySet([(0,), (1,), (2,), (3,), (4,)], domain.values_by_col)

        canon_queries = naive_bayes_3_way_cross_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = naive_bayes_3_way_cross_queries_missing.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = naive_bayes_cross_queries.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = one_way_marginals.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(n_canon_queries, 1 + 1 + 2 + 3 + 4 + 5)
        self.assertEqual(rank, n_canon_queries + 1)

        canon_queries = one_way_marginals_missing.get_canonical_queries().flatten()
        n_canon_queries = len(canon_queries.queries)
        rank = self.query_matrix_rank(domain, canon_queries)
        self.assertEqual(n_canon_queries, 1 + 1 + 2 + 3 + 4)
        self.assertEqual(rank, n_canon_queries + 1)


if __name__ == "__main__":
    unittest.main()
