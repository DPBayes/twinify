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
from typing import Optional, Iterator, Iterable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.infer.util as nummcmc_util
from jax import random
import twinify.napsu_mq.maximum_entropy_model as mem


def run_numpyro_mcmc(
        rng: random.PRNGKey, suff_stat, n, sigma_DP, max_ent_dist, prior_mu=0, prior_sigma=10,
        num_samples=1000, num_warmup=500, num_chains=1, disable_progressbar=False,
):
    kernel = numpyro.infer.NUTS(model=mem.normal_prior_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential"
    )
    mcmc.run(rng, suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    return mcmc


def run_numpyro_mcmc_normalised(
        rng: random.PRNGKey, suff_stat, n, sigma_DP, max_ent_dist, laplace_approx, prior_sigma=10,
        num_samples=1000, num_warmup=500, num_chains=1, disable_progressbar=False,
):
    kernel = numpyro.infer.NUTS(model=mem.normal_prior_normalised_model_numpyro, max_tree_depth=12)
    mcmc = numpyro.infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=not disable_progressbar, jit_model_args=False, chain_method="sequential"
    )

    mean_guess = jnp.array(laplace_approx.mean.detach().numpy())
    L_guess = jnp.linalg.cholesky(jnp.array(laplace_approx.covariance_matrix.detach().numpy()))
    mcmc.run(rng, jnp.array(suff_stat), n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess)

    def backtransform(lambdas):
        return (L_guess @ lambdas.transpose()).transpose() + mean_guess

    return mcmc, backtransform


def run_numpyro_laplace_approximation(
        rng: random.PRNGKey, suff_stat, n, sigma_DP, max_ent_dist, prior_mu=0, prior_sigma=10
):
    init_lambdas, potential_fn, t, mt = nummcmc_util.initialize_model(
        rng, mem.normal_prior_model_numpyro, model_args=(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    )
    lambdas = init_lambdas[0]["lambdas"]
    result = jax.scipy.optimize.minimize(lambda l: potential_fn({"lambdas": l}), lambdas, method="BFGS", tol=1e-2)
    mean = result.x
    prec = jax.hessian(lambda l: potential_fn({"lambdas": l}))(mean)
    laplace_approx = numpyro.distributions.MultivariateNormal(loc=mean, precision_matrix=prec)
    return laplace_approx, result.success