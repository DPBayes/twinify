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
from typing import Iterable, Tuple
import numpy as np

def eps_delta_budget_to_rho_budget(epsilon: float, delta: float) -> float:
    """Convert (epsilon, delta)-DP to rho-zCDP.
    Args:
        epsilon (float)
        delta (float)
    Returns:
        float: rho
    """
    beta = np.sqrt(-np.log(delta))
    return (np.sqrt(beta ** 2 + epsilon) - beta) ** 2


def gauss_mechanism_sigma(rho: float, sensitivity: float) -> float:
    """Compute noise standard deviation for Gaussian mechanism with zCDP.
    Args:
        rho (float): zCDP privacy bound.
        sensitivity (float): Sensitivity of the Gaussian mechanism.
    Returns:
        float: Required noise standard deviation.
    """
    return np.sqrt(sensitivity ** 2 / (2 * rho))


def gauss_mechanism_composition_sigma(rho: float, sensitivity_counts: Iterable[Tuple[float, int]]) -> float:
    """Compute noise standard deviation for a composition of Gaussian mechanism with zCDP.
    Args:
        rho (float): zCDP privacy bound.
        sensitivity_counts (list((float, int))): List of pairs (sensitivity, number of queries).
    Returns:
        float: Required noise standard deviation.
    """
    return np.sqrt(np.sum(np.array([sc[1] * sc[0] ** 2 for sc in sensitivity_counts])) / (2 * rho))


def gauss_mechanism(x, rho, sensitivity) -> Tuple[float, float]:
    """Run the Gaussian mechanism.
    Args:
        x (float): Value to release.
        rho (float): zCDP privacy parameter.
        sensitivity (float): Sensitivity of x.
    Returns:
        float: Noisy value of x.
    """
    sigma = gauss_mechanism_sigma(rho, sensitivity)
    return gauss_mechanism_with_sigma(x, sigma), sigma


def gauss_mechanism_with_sigma(x: float, sigma: float) -> float:
    """Run the Gaussian mechanism with given noise standard deviation.
    Args:
        x (float): Value to release.
        sigma (float): Noise standard deviation.
    Returns:
        float: Noisy value of x.
    """
    return np.random.normal(loc=x, scale=sigma)


def report_noisy_max(x, rho: float, sensitivity: float) -> np.ndarray[int]:
    """Report noisy max with Gumbel noise.
    Args:
        x (torch.tensor): Input data.
        rho (float): zCDP privacy parameter.
        sensitivity (float): Sensitivity of x.
    Returns:
        _type_: _description_
    """
    noise_scale = sensitivity / np.sqrt(2 * rho)
    return np.argmax(x + np.random.gumbel(loc=0, scale=noise_scale, size=x.shape))
