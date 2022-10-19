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
from typing import Dict, Tuple, Iterable

import itertools
from functools import reduce
from operator import mul

import jax.random
import numpy as np
from scipy.special import logsumexp
from twinify.napsu_mq.marginal_query import QueryList


class MaximumEntropyDistribution:
    """Implementation of MED without using graphical models.
    This class computes all quantities naively, so it is only useful for testing
    with small output domains.
    """

    def __init__(self, values_by_feature: Dict, queries: QueryList):
        """Create The MaximumEntropyDistribution.
        Args:
            values_by_feature (dict): A dict of the possible values for each variable.
            queries (MarginalQueryList): The queries forming the sufficient statistic.
        """
        self.values_by_feature = values_by_feature
        self.d = len(values_by_feature.keys())
        self.suff_stat_d = len(queries.queries)
        self.lambda_d = self.suff_stat_d
        self.queries = queries

        self.x_values = self.get_x_values(list(values_by_feature.keys()))
        self.compute_suff_stat_cache()

    def lambda0(self, lambdas: np.ndarray) -> np.ndarray:
        result = logsumexp(self.suff_stat_array @ lambdas + self.suff_stat_log_count_array, axis=0)
        return result

    def suff_stat_mean_and_lambda0(self, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lambda0: np.ndarray = self.lambda0(lambdas)
        sum_term: np.ndarray = np.log(self.suff_stat_array) + \
                               (self.suff_stat_array @ lambdas).reshape(-1, 1) + \
                               self.suff_stat_log_count_array.reshape(-1, 1)
        suff_stat_mean = np.sum(np.exp(sum_term - lambda0), axis=0)
        return suff_stat_mean, lambda0

    def suff_stat_mean(self, lambdas: np.ndarray) -> np.ndarray:
        return self.suff_stat_mean_and_lambda0(lambdas)[0]

    def suff_stat_cov(self, lambdas: np.ndarray) -> np.ndarray:
        return self.suff_stat_mean_and_cov_explicit(lambdas)[1]

    def suff_stat_mean_and_cov(self, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(lambdas.shape) > 1:
            n, d = lambdas.shape
            mean = np.zeros(lambdas.shape)
            cov = np.zeros(n, d, d)
            for i in range(n):
                mean_l, cov_l = self.suff_stat_mean_and_cov_explicit(lambdas[i, :])
                mean[i, :] = mean_l
                cov[i, :, :] = cov_l
            return mean, cov

        else:
            return self.suff_stat_mean_and_cov_explicit(lambdas)

    def sample_inds(self, rng_key: jax.random.PRNGKey, lambdas: np.ndarray, n: int = 1) -> np.ndarray:
        random_values = jax.random.uniform(key=rng_key, shape=[n], minval=0, maxval=1).reshape(-1, 1)
        thresholds = self.pmf_all_values(lambdas).cumsum(0)
        inds = np.argmax((random_values < thresholds).astype(int), axis=1).reshape(-1)
        return inds

    def sample(self, rng_key: jax.random.PRNGKey, lambdas: np.ndarray, n: int = 1) -> np.ndarray:
        return self.x_values[self.sample_inds(rng_key, lambdas, n), :]

    def compute_suff_stat_cache(self) -> None:
        suff_stats_all_int = self.queries(self.x_values)
        self.suff_stats_all = suff_stats_all_int.astype(np.double)

        self.suff_stat_array, self.suff_stat_count_array = np.unique(suff_stats_all_int, return_counts=True, axis=0)
        self.suff_stat_array = self.suff_stat_array.astype(np.double)
        self.suff_stat_count_array = self.suff_stat_count_array.astype(np.double)

        self.suff_stat_log_count_array = np.log(self.suff_stat_count_array)

    def get_x_values(self, variables: Iterable) -> np.ndarray:
        x_val_count = reduce(mul, (len(self.values_by_feature[var]) for var in variables), 1)
        x_values_all = np.zeros((x_val_count, self.d), dtype=np.int64)

        for i, val in enumerate(itertools.product(*[self.values_by_feature[var] for var in variables])):
            x_values_all[i, variables] = np.array(val, dtype=np.int64)
        return x_values_all

    def suff_stat_mean_and_cov_explicit(self, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = lambdas.shape[0]
        mean, lambda0 = self.suff_stat_mean_and_lambda0(lambdas)
        bmm_term = self.suff_stat_array.reshape(-1, d, 1) @ self.suff_stat_array.reshape(-1, 1, d)
        sum_term = np.log(bmm_term) + (self.suff_stat_array @ lambdas).reshape(
            (-1, 1, 1)) + self.suff_stat_log_count_array.reshape((-1, 1, 1))
        cov = np.sum(np.exp(sum_term - lambda0), axis=0) - np.outer(mean, mean)
        # Ensure returned matrix is positive-definite
        return mean, cov + np.eye(lambdas.shape[0]) * 1e-12

    def pmf_all_values(self, lambdas: np.ndarray) -> np.ndarray:
        return np.exp(self.suff_stats_all @ lambdas - self.lambda0(lambdas))

    def mean_query_values(self, queries: QueryList, lambdas: np.ndarray) -> np.ndarray:
        query_values = queries(self.x_values)
        pmf_all_values = self.pmf_all_values(lambdas)
        pmf = pmf_all_values.reshape(-1, 1)
        return (pmf * query_values).sum(axis=0)

    def conjugate_unnorm_logpdf(self, lambdas: np.ndarray, chi: np.ndarray, nu: np.ndarray) -> np.ndarray:
        lambda0 = self.lambda0(lambdas)
        return np.dot(chi, lambdas) - nu * lambda0
