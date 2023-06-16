# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023- twinify Developers and their Assignees

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
import pandas as pd

import d3p.random

from twinify.napsu_mq import NapsuMQModel

class GithubBugTests(unittest.TestCase):

    def test_issue_50(self):
        n = 4

        data = pd.DataFrame({
            'A': np.random.randint(500, 1000, size=n),
            'B': np.random.randint(500, 1000, size=n),
            'C': np.random.randint(500, 1000, size=n)
        }, dtype='category')

        data['A'] = data['A'].astype(int)

        rng = d3p.random.PRNGKey(42)

        model = NapsuMQModel(required_marginals=[])
        result = model.fit(data=data, rng=rng, epsilon=1, delta=(n ** (-2)),
                        use_laplace_approximation=True)
        
        for k in data:
            self.assertEqual(list(np.unique(data[k])), list(result.dataframe_domain[k])) # currently fails: dataframedata.values_by_cols is overwritten at some point in NapsuMqModel.fit ?

        # also test that synthetic data is properly mapped
        
        
if __name__ == '__main__':
    unittest.main()