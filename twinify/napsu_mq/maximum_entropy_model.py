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

from typing import Union

import numpyro
import numpyro.distributions as numdist
import jax.numpy as jnp
from twinify.napsu_mq.markov_network import MarkovNetwork


def normal_prior_model_numpyro(dp_suff_stat: jnp.ndarray, n: int, sigma_DP: float, prior_mu: Union[float, jnp.ndarray],
                               prior_sigma: float, med: MarkovNetwork):
    """NumPyro model for NAPSU-MQ with isotropic Gaussian prior.
    Args:
        dp_suff_stat (jax.ndarray): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_mu (float or jax.ndarray): Prior mean.
        prior_sigma (float): Prior standard deviation.
        med (MarkovNetwork): An implementation of MED.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    lambdas = numpyro.sample("lambdas", numdist.MultivariateNormal(jnp.ones(lambda_d) * prior_mu,
                                                                   covariance_matrix=jnp.eye(
                                                                       lambda_d) * prior_sigma ** 2))
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return numpyro.sample("a_hat",
                          numdist.MultivariateNormal(n * mean, covariance_matrix=n * cov + jnp.eye(d) * sigma_DP ** 2),
                          obs=jnp.array(dp_suff_stat))


def normal_prior_normalised_model_numpyro(dp_suff_stat: jnp.ndarray, n: int, sigma_DP: float, prior_sigma: float,
                                          med: MarkovNetwork, mean_guess: jnp.ndarray, L_guess: jnp.ndarray):
    """NumPyro model for NAPSU-MQ with isotropic Gaussian prior and normalisation.
    Args:
        dp_suff_stat (jax.ndarray): Noisy sufficient statistic.
        n (int): Number of datapoints.
        sigma_DP (float): Noise standard deviation.
        prior_sigma (float): Prior standard deviation.
        med (MarkovNetwork): An implementation of MED.
        mean_guess (jax.ndarray): A guess for the posterior mean.
        L_guess (jax.ndarray): A guess for the Cholesky decomposition of posterior covariance.
    """
    d = med.suff_stat_d
    lambda_d = med.lambda_d

    prior_cov_inner = jnp.linalg.solve(L_guess, jnp.eye(lambda_d) * prior_sigma ** 2)
    norm_prior_cov = jnp.linalg.solve(L_guess, prior_cov_inner.transpose()).transpose()

    norm_prior_mu = jnp.linalg.solve(L_guess, -mean_guess.reshape((-1, 1))).flatten()

    norm_lambdas = numpyro.sample("norm_lambdas",
                                  numdist.MultivariateNormal(norm_prior_mu, covariance_matrix=norm_prior_cov))
    lambdas = L_guess @ norm_lambdas + mean_guess
    mean, cov = med.suff_stat_mean_and_cov(lambdas)
    return numpyro.sample("a_hat",
                          numdist.MultivariateNormal(n * mean, covariance_matrix=n * cov + jnp.eye(d) * sigma_DP ** 2),
                          obs=dp_suff_stat)
