# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings as _warnings
from jax.config import config as _config
_config.update("jax_enable_x64", True)
import jax.numpy as _jnp
if _jnp.array([1.]).dtype != _jnp.float64:
    _warnings.warn(
        "Failed to set floating point precision to 64 bits. You may experience instabilities "
        "with twinify as a result.\n"
        "This can happen when the jax library is already initialized in your scripts before twinify is loaded. "
        "In that case, please enable 64 bit precision by including the following before the first usage of jax:\n"
        "   from jax.config import config\n"
        "   config.update('jax_enable_x64', True).",
        stacklevel=2
    )

import twinify.dpvi
import twinify.napsu_mq
from twinify.dataframe_data import DataDescription
from twinify.version import VERSION
__version__ = VERSION
from twinify.base import InferenceModel, InferenceResult
