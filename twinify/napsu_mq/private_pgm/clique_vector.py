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

# Originally from https://github.com/ryan112358/private-pgm/blob/557c077708d3559212a8f65dff3eccd3fd244abb/src/mbi/clique_vector.py
# Modified by the twinify Developers under the Apache 2.0 license
# Changed import statements

import numpy as np
from twinify.napsu_mq.private_pgm.factor import Factor


class CliqueVector(dict):
    """ This is a convenience class for simplifying arithmetic over the
        concatenated vector of marginals and potentials.
        These vectors are represented as a dictionary mapping cliques (subsets of attributes)
        to marginals/potentials (Factor objects)
    """

    def __init__(self, dictionary):
        self.dictionary = dictionary
        dict.__init__(self, dictionary)

    @staticmethod
    def zeros(domain, cliques):
        return CliqueVector({cl: Factor.zeros(domain.project(cl)) for cl in cliques})

    @staticmethod
    def ones(domain, cliques):
        return CliqueVector({cl: Factor.ones(domain.project(cl)) for cl in cliques})

    @staticmethod
    def uniform(domain, cliques):
        return CliqueVector({cl: Factor.uniform(domain.project(cl)) for cl in cliques})

    @staticmethod
    def random(domain, cliques, prng=np.random):
        return CliqueVector({cl: Factor.random(domain.project(cl), prng) for cl in cliques})

    @staticmethod
    def normal(domain, cliques, prng=np.random):
        return CliqueVector({cl: Factor.normal(domain.project(cl), prng) for cl in cliques})

    @staticmethod
    def from_data(data, cliques):
        ans = {}
        for cl in cliques:
            mu = data.project(cl)
            ans[cl] = Factor(mu.domain, mu.datavector())
        return CliqueVector(ans)

    def combine(self, other):
        # combines this CliqueVector with other, even if they do not share the same set of factors
        # used for warm-starting optimization
        # Important note: if other contains factors not defined within this CliqueVector, they
        # are ignored and *not* combined into this CliqueVector
        for cl in other:
            for cl2 in self:
                if set(cl) <= set(cl2):
                    self[cl2] += other[cl]
                    break

    def __mul__(self, const):
        ans = {cl: const * self[cl] for cl in self}
        return CliqueVector(ans)

    def __rmul__(self, const):
        return self.__mul__(const)

    def __add__(self, other):
        if np.isscalar(other):
            ans = {cl: self[cl] + other for cl in self}
        else:
            ans = {cl: self[cl] + other[cl] for cl in self}
        return CliqueVector(ans)

    def __sub__(self, other):
        return self + -1 * other

    def exp(self):
        ans = {cl: self[cl].exp() for cl in self}
        return CliqueVector(ans)

    def log(self):
        ans = {cl: self[cl].log() for cl in self}
        return CliqueVector(ans)

    def dot(self, other):
        return sum((self[cl] * other[cl]).sum() for cl in self)

    def size(self):
        return sum(self[cl].domain.size() for cl in self)

