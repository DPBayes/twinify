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

from typing import Optional
import jax.numpy as jnp
import jax

def slice_feature(
        x: Optional[jnp.ndarray],
        feature_idx_start: int,
        feature_idx_limit: Optional[int] = None,
        stride: int = 1,
        dtype: Optional[jnp.dtype] = None
    ):
    """
    Returns a slice over the feature dimension (axis 1) of a data set. Returns
    `None` without failure if `None` is input.

    If `feature_idx_limit` is `None` (and `x` is an array), returns all columns starting
    from  `feature_idx_start`, i.e., x[:, feature_idx_start::stride]`.

    If `feature_idx_limit` is an integer, returns the slice indicated by the arguments, i.e.,
    `x[:, feature_idx_start:feature_idx_limit:stride]`.

    Args:
        x (jax.numpy.ndarray, None): Array or None.
        feature_idx_start (int) : Feature index of where to start the slice.
        feature_idx_limit (int, None): Optional feature index where to stop the slice.
        stride (int): Optional stride, defaults to 1.
        dtype (jnp.dtype, None): Optional output dtype to convert the slice to.
    """
    if x is None:
        return None

    if dtype is None:
        dtype = jnp.dtype(x)

    return jax.lax.slice_in_dim(x, feature_idx_start, feature_idx_limit, stride, axis=1).astype(dtype)
