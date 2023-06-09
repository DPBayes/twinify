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

from typing import Union, Callable, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.infer.util as nummcmc_util
from jax import random
import twinify.napsu_mq.maximum_entropy_model as mem
from twinify.napsu_mq.markov_network import MarkovNetwork


def run_numpyro_mcmc(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetwork,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, num_samples: int = 1000,
        num_warmup: int = 500, num_chains: int = 1, show_progressbar: bool = True,
) -> numpyro.infer.MCMC:
    """Run MCMC inference (NUTS) with Numpyro on maximum entropy distribution with multivariate normal prior

    Args:
        rng (jax.random.PRNGKey): Jax random key for MCMC
        suff_stat (jax.numpy.ndarray): Noisy sufficient statistics array with DP noise added
        n (int): Number of datapoints
        sigma_DP (float): Noise standard deviation
        max_ent_dist (MarkovNetwork): Markov network representation of maximum entropy distribution
        prior_mu (float or jax.numpy.ndarray): Mean prior for multivariate normal distribution
        prior_sigma (float): Standard deviation prior for multivariate normal distribution
        num_samples (int): Number of samples for MCMC
        num_warmup (int): Number of warm-up (burn-in) samples for MCMC
        num_chains (int): Number of chains in MCMC
        disable_progressbar (bool): Disable progressbar for MCMC

    Returns:
        mcmc: Numpyro inference MCMC object
    """
    kernel = numpyro.infer.NUTS(model=mem.normal_prior_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=show_progressbar, jit_model_args=False, chain_method="sequential"
    )
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    return mcmc


def run_numpyro_mcmc_normalised(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetwork,
        laplace_approx: numpyro.distributions.MultivariateNormal, prior_sigma: float = 10,
        num_samples: int = 1000, num_warmup: int = 500, num_chains: int = 1, show_progressbar: bool = True,
) -> Tuple[numpyro.infer.MCMC, Callable]:
    """Run MCMC inference (NUTS) with Numpyro on maximum entropy distribution with normalized multivariate normal prior

    Args:
        rng (jax.random.PRNGKey): Jax random key for MCMC
        suff_stat (jax.numpy.ndarray): Noisy sufficient statistics array with DP noise added
        n (int): Number of datapoints
        sigma_DP (float): Noise standard deviation
        max_ent_dist (MarkovNetwork): Markov network representation of maximum entropy distribution
        laplace_approx (numpyro.distributions.MultivariateNormal): Laplace approximation of multivariate normal for MCMC
        prior_sigma (float): Standard deviation prior for multivariate normal distribution
        num_samples (int): Number of samples for MCMC
        num_warmup (int): Number of warm-up (burn-in) samples for MCMC
        num_chains (int): Number of chains in MCMC
        disable_progressbar (bool): Disable progressbar for MCMC

    Returns:
        mcmc: Numpyro inference MCMC object
        backtransform: Function to transform posterior values back to non-normalized space
    """
    kernel = numpyro.infer.NUTS(model=mem.normal_prior_normalised_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=show_progressbar, jit_model_args=False, chain_method="sequential"
    )

    mean_guess = laplace_approx.mean
    L_guess = jnp.linalg.cholesky(laplace_approx.covariance_matrix)
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess)

    def backtransform(lambdas: jnp.ndarray) -> jnp.ndarray:
        return (L_guess @ lambdas.transpose()).transpose() + mean_guess

    return mcmc, backtransform


class ConvergenceException(Exception):
    """Convergence error in optimization process"""


def run_numpyro_laplace_approximation(
        rng: random.PRNGKey, suff_stat: jnp.ndarray, n: int, sigma_DP: float, max_ent_dist: MarkovNetwork,
        prior_mu: Union[float, jnp.ndarray] = 0, prior_sigma: float = 10, 
        max_retries=5, tol=1e-2, max_iters=100
) -> Tuple[numpyro.distributions.MultivariateNormal, bool]:
    """Run Laplace approximation on the maximum entropy distribution

    Args:
        rng (jax.random.PRNGKey): Jax random key for MCMC
        suff_stat (jax.numpy.ndarray): Noisy sufficient statistics array with DP noise added
        n (int): Number of datapoints
        sigma_DP (float): Noise standard deviation
        max_ent_dist (MarkovNetwork): Markov network representation of maximum entropy distribution
        prior_mu (float or jax.numpy.ndarray): Mean prior for multivariate normal distribution
        prior_sigma (float): Standard deviation prior for multivariate normal distribution
        max_retries (int): Times to retry the approximation

    Returns:
        laplace_approx: Laplace approximation for the maximum entropy distribution
        result.success: Boolean value if approximation was successful
    """

    key, *subkeys = random.split(rng, max_retries + 1)
    fail_count = 0

    for i in range(0, max_retries + 1):

        rng = subkeys[i]

        init_lambdas, potential_fn, t, mt = nummcmc_util.initialize_model(
            rng, mem.normal_prior_model_numpyro,
            model_args=(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
        )

        lambdas = init_lambdas[0]["lambdas"]

        result = jax.scipy.optimize.minimize(lambda l: potential_fn({"lambdas": l}), lambdas, method="BFGS", tol=tol, options={"maxiter": max_iters})
        if not result.success:
            fail_count += 1
        else:
            mean = result.x
            break

        if fail_count == max_retries:
            raise ConvergenceException(f"Minimize function failed to converge with {max_retries} retries")

    prec = jax.hessian(lambda l: potential_fn({"lambdas": l}))(mean)
    laplace_approx = numpyro.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    return laplace_approx, result.success
