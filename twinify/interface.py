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
from numpyro.primitives import deterministic

__all__ = ['get_feature', 'sample_combined']

def get_feature(x, i):
    """
    Wraps access to a feature value of an observation and deals gracefully
    if no observation is given.

    Args:
        x (jax.numpy.ndarray): Array holding all features of a single data instance or None.
        i (int): Index of feature value to return.
    Returns:
        x[i] if x is not None; otherwise None
    """
    return None if x is None else np.take(x, i, -1)

def sample_combined(*feature_samples):
    """
    Combines individual feature samples into a single sample of the full
    observation vector expected by twinify.

    Args:
        *feature_samples: Iterable of individual feature values.
    Samples:
        site `x` holding the combined feature vector
    Returns:
        jax.numpy.ndarray: sampled values
    """
    return deterministic('x', np.stack(feature_samples, axis=1))
