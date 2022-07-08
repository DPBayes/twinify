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


from itertools import chain, combinations
from typing import Iterable, List


def powerset(iterable: Iterable) -> List:
    """Create powerset from iterable
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    source: https://docs.python.org/3/library/itertools.html#itertools-recipes

    Args:
        iterable (Iterable): Iterable to create powerset from

    Returns:
        list (List): Powerset from iterable
    """
    s = list(iterable)
    return [set(t) for t in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))]