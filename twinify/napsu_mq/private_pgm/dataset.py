# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © Ryan McKenna, © 2022- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Originally from https://github.com/ryan112358/private-pgm/blob/557c077708d3559212a8f65dff3eccd3fd244abb/src/mbi/dataset.py
# Modified by the twinify Developers under the Apache 2.0 license
# Changed import statements

import numpy as np
import pandas as pd
import json
from twinify.napsu_mq.private_pgm.domain import Domain

class Dataset:
    def __init__(self, df, domain, weights=None):
        """ create a Dataset object
        :param df: a pandas dataframe
        :param domain: a domain object
        :param weight: weight for each row
        """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        assert weights is None or df.shape[0] == weights.size
        self.domain = domain
        self.df = df.loc[:, domain.attrs]
        self.weights = weights

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain
        :param domain: The domain object
        :param N: the number of individuals
        """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain):
        """ Load data into a dataset object
        :param path: path to csv file
        :param domain: path to json file encoding the domain information
        """
        df = pd.read_csv(path)
        config = json.load(open(domain))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)

    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain, self.weights)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.df.shape[0]

    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=self.weights)[0]
        return ans.flatten() if flatten else ans
