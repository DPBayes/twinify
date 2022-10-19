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
import twinify.napsu_mq.rubins_rules as rubins_rules


class RubinsRulesTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(643739)
        self.synthetic_data_sets = [np.random.normal(loc=10, scale=1, size=1000) for _ in range(10)]

    def test_confidence_interval(self):
        means = [np.mean(set) for set in self.synthetic_data_sets]
        variances = [np.var(set, axis=0) / len(set) for set in self.synthetic_data_sets]

        conf_intervals = rubins_rules.conf_int(means, variances, 0.95)

        self.assertAlmostEqual(float(np.mean(conf_intervals)), 10, delta=0.5)
        self.assertAlmostEqual(conf_intervals[0], 5, delta=0.5)
        self.assertAlmostEqual(conf_intervals[1], 15, delta=0.5)

    def test_non_negative_confidence_interval(self):
        means = np.mean(self.synthetic_data, axis=1)
        variances = np.var(self.synthetic_data, axis=1)

        conf_intervals = rubins_rules.non_negative_conf_int(means, variances, 0.95, n=100 * 1000,
                                                               n_orig=1000)

        self.assertAlmostEqual(float(np.mean(conf_intervals)), 500, delta=1)

