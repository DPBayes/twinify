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
from typing import Optional, Union, Tuple, Any
import warnings
import numpy as np
from scipy import special
import scipy.optimize as optim


def delta(epsilon: float, sens_per_sigma: float) -> float:
    """Compute delta for given epsilon and sensitivity per noise standard deviation for the Gaussian mechanism.
    Args:
        epsilon (float)
        sens_per_sigma (float): Sensitivity per noise standard deviation.
    Returns:
        float: Delta
    """
    if sens_per_sigma < 0:
        warnings.warn(
            f"Sensitivity per sigma was negative: {sens_per_sigma}. "
            f"Sensitivity per sigma should be non-negative. Defaulting to 0"
        )

    if sens_per_sigma <= 0:
        return 0
    mu = sens_per_sigma ** 2 / 2
    term1 = special.erfc((epsilon - mu) / np.sqrt(mu) / 2)
    term2 = np.exp(epsilon) * special.erfc((epsilon + mu) / np.sqrt(mu) / 2)
    return 0.5 * (term1 - term2)


def find_sens_per_sigma(epsilon: float, delta_bound: float, sens_per_sigma_upper_bound: Optional[float] = 20) -> Union[
    float, Tuple[float, Any]]:
    """Find the required sensitivity per noise standard deviation for (epsilon, delta)-DP with Gaussian mechanism.
    Args:
        epsilon (float)
        delta_bound (float)
        sens_per_sigma_upper_bound (float, optional): Upper bound guess on sensitivity per sigma. Defaults to 20.
    Returns:
        float: The required sensitivity per noise standard deviation.
    """
    return optim.brentq(lambda sigma: delta(epsilon, sigma) - delta_bound, 0, sens_per_sigma_upper_bound)


def sigma(epsilon: float, delta_bound: float, sensitivity: float, sigma_upper_bound: Optional[float] = 20) -> float:
    """Find the required noise standard deviation for the Gaussian mechanism with (epsilon, delta)-DP.
    Args:
        epsilon (float)
        delta_bound (float)
        sensitivity (float): Sensitivity of the Gaussian mechanism.
        sigma_upper_bound (float, optional): Guess for an upper bound on sensitivity / sigma. Defaults to 20.
    Returns:
        float: The required noise standard deviation.
    """
    return sensitivity / find_sens_per_sigma(epsilon, delta_bound, sigma_upper_bound)
