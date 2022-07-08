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
from typing import Tuple, Optional

import numpy as np
import scipy.stats
import scipy.stats as stats


def compute_aggregates(q: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aggregates for Rubin's rules from point and variance estimates.
    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
    Returns:
        tuple: (q_m, b_m, u_m).
    """
    q_m = np.mean(q)
    b_m = np.var(q, ddof=1)
    u_m = np.mean(u)
    return q_m, b_m, u_m


def conf_int(q: np.ndarray, u: np.ndarray, conf_level: float) -> np.ndarray:
    """Compute confidence interval with Rubin's rules.
    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        conf_level (float): Confidence level.
    Returns:
        ndarray: The confidence interval.
    """
    dist = conf_int_distribution(q, u)
    return dist.interval(conf_level) if dist is not None else np.repeat(np.nan, 2)


def non_negative_conf_int(q: np.ndarray, u: np.ndarray, conf_level: float, n: int, n_orig: int) -> np.ndarray:
    """Compute confidence interval with Rubin's rules and non-negative variance estimate.
    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        conf_level (float): Confidence level.
        n (int): Number of synthetic datapoints.
        n_orig (int): Number of original datapoints.
    Returns:
        ndarray: The confidence interval.
    """
    dist = conf_int_distribution(q, u, True, n, n_orig)
    return dist.interval(conf_level) if dist is not None else np.repeat(np.nan, 2)


def conf_int_distribution(q: np.ndarray, u: np.ndarray, use_nonnegative_variance: Optional[bool] = False,
                          n: Optional[int] = None, n_orig: Optional[int] = None) -> Optional[scipy.stats.t_gen]:
    """Compute the estimator distribution with Rubin's rules used for confidence intervals and hypothesis tests.
    Args:
        q (ndarray): Point estimates.
        u (ndarray): Variance estimates.
        use_nonnegative_variance (bool, optional): Use the non-negative variance estimate. Defaults to False.
        n (int, optional): Number of synthetic datapoints. Required with non-negative variance estimate. Defaults to None.
        n_orig (int, optional): Number of original datapoints. Required with non-negative variance estimate. Defaults to None.
    Returns:
        scipy distribution: The distribution as a scipy.stats distribution object.
    """
    q_m, b_m, u_m = compute_aggregates(q, u)
    m = q.size
    T_m = (1 + 1 / m) * b_m - u_m
    if use_nonnegative_variance:
        T_m = T_m if T_m > 0 else n / n_orig * u_m
    degree = (m - 1) * (1 - u_m / ((1 + 1 / m) * b_m)) ** 2
    if not np.isfinite(degree):
        print("m: {}, u_m: {}, b_m: {}".format(m, u_m, b_m))

    if T_m < 0:
        return None
    else:
        return stats.t(loc=q_m, scale=np.sqrt(T_m), df=degree)
