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

from twinify.napsu_mq.utils import powerset


class PowerSetTest(unittest.TestCase):

    def test_powerset(self):
        powerset_result = powerset([1, 2, 3])
        expected_results = [set(), {1, }, {2, }, {3, }, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}]
        self.assertEqual(powerset_result, expected_results)


if __name__ == "__main__":
    unittest.main()
