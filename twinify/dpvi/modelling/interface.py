# Copyright 2020 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper routines for modelling.
"""

import jax.numpy as np

__all__ = ['get_feature']

def get_feature(x, idxs): # TODO: fix and test, or remove
    """
    Wraps access to a feature value of an observation and deals gracefully
    if no observation is given.

    Args:
        x (jax.numpy.ndarray): Array holding all features of a single data instance or None.
        idxs : Indices or index slice of feature value to return.
    Returns:
        x[i] if x is not None; otherwise None
    """
    return None if x is None else x[:, idxs]
