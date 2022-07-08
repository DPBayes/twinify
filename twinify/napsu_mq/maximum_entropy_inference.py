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
import numpy as np
import numpyro
import numpyro.infer.util as nummcmc_util
import pyro.infer.mcmc.util as mcmc_util
import torch
import torch.autograd as autograd
import torch.optim as optim
import tqdm
from jax import random
from pyro.infer import MCMC, NUTS
import twinify.napsu_mq.maximum_entropy_model as mem


def rng_state_set(generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
    if generator is not None:
        old_rng_state = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
        return old_rng_state
    return None


def rng_state_restore(old_rng_state: torch.Tensor) -> None:
    if old_rng_state is not None:
        torch.set_rng_state(old_rng_state)


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


def run_mcmc(
        suff_stat, n, sigma_DP, max_ent_dist,
        prior_mu=0, prior_sigma=10,
        num_samples=2000, warmup_steps=200, num_chains=4,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(mem.normal_prior_model, jit_compile=False)

    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=num_chains,
        mp_context="forkserver", disable_progbar=disable_progressbar
    )
    mcmc.run(suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
    rng_state_restore(ors)
    return mcmc


def run_mcmc_normalised(
        suff_stat, n, sigma_DP, max_ent_dist,
        prior_mu=0, prior_sigma=10,
        num_samples=2000, warmup_steps=200, num_chains=4,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(mem.normal_prior_normalised_model, jit_compile=False)
    laplace_approx, losses, fail_count = laplace_approximation_normal_prior(suff_stat, n, sigma_DP, max_ent_dist,
                                                                            prior_mu, prior_sigma)
    mean_guess = laplace_approx.loc
    L_guess = torch.linalg.cholesky(laplace_approx.covariance_matrix)

    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=num_chains,
        mp_context="forkserver", disable_progbar=disable_progressbar
    )
    mcmc.run(suff_stat, n, sigma_DP, prior_sigma, max_ent_dist, mean_guess, L_guess)

    def backtransform(lambdas):
        return (L_guess.detach().numpy() @ lambdas.T).T + mean_guess.detach().numpy()

    rng_state_restore(ors)
    return mcmc, backtransform


def run_mvnormal_prior_mcmc(
        suff_stat, n, sigma_DP, max_ent_dist,
        prior_mu=0, prior_cov=10,
        num_samples=2000, warmup_steps=200, num_chains=4,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(mem.mvnormal_prior_model, jit_compile=False)
    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=num_chains,
        mp_context="forkserver", disable_progbar=disable_progressbar
    )
    mcmc.run(suff_stat, n, sigma_DP, prior_mu, prior_cov, max_ent_dist)
    rng_state_restore(ors)
    return mcmc


def run_conjugate_prior_mcmc(
        suff_stat, n, sigma_DP, max_ent_dist,
        prior_chi, prior_nu,
        num_samples=2000, warmup_steps=200,
        disable_progressbar=False, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    nuts_kernel = NUTS(
        potential_fn=lambda l: mem.conjugate_prior_potential(l[0], suff_stat, n, sigma_DP, prior_chi, prior_nu,
                                                             max_ent_dist), jit_compile=False)

    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples,
        warmup_steps=warmup_steps, num_chains=1,
        mp_context="forkserver", disable_progbar=disable_progressbar,
        initial_params={0: torch.zeros(suff_stat.shape[0])}
    )
    mcmc.run()
    rng_state_restore(ors)
    return mcmc


def laplace_optimisation(lambdas, potential_fn, max_iters, tol, max_loss_jump):
    opt = optim.LBFGS([lambdas])
    losses = []
    for i in range(max_iters):
        def closure():
            opt.zero_grad()
            output = potential_fn({"lambdas": lambdas})
            losses.append(output.item())
            output.backward()
            return output

        opt.step(closure)
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tol:
            return True, lambdas, losses
        if len(losses) > 1 and (losses[-1] - losses[-2]) > max_loss_jump:
            return False, lambdas, losses

    return False, lambdas, losses


def laplace_approximation_normal_prior(
        suff_stat, n, sigma_DP, max_ent_dist, prior_mu=0, prior_sigma=10, max_iters=500,
        tol=1e-5, max_loss_jump=1e3, max_retries=2, generator: Optional[torch.Generator] = None
):
    ors = rng_state_set(generator)

    fail_count = 0
    for i in range(max_retries + 1):
        init_lambdas, potential_fn, t, mt = mcmc_util.initialize_model(
            mem.normal_prior_model, (suff_stat, n, sigma_DP, prior_mu, prior_sigma, max_ent_dist)
        )
        lambdas = init_lambdas["lambdas"].clone().requires_grad_(True)
        success, lambdas, losses = laplace_optimisation(lambdas, potential_fn, max_iters, tol, max_loss_jump)
        if success:
            break
        else:
            fail_count += 1

    hess = autograd.functional.hessian(lambda l: potential_fn({"lambdas": l}), lambdas)
    laplace_approx = torch.distributions.MultivariateNormal(lambdas, precision_matrix=hess)

    rng_state_restore(ors)

    return laplace_approx, losses, fail_count


def laplace_approximation_conjugate_prior(
        suff_stat, n, sigma_DP, max_ent_dist, init_lambdas, prior_chi, prior_nu, max_iters=500,
        tol=1e-5, max_loss_jump=1e3, max_retries=2, generator=Optional[torch.Generator]
):
    ors = rng_state_set(generator)
    potential_fn = lambda l: mem.conjugate_prior_potential(l["lambdas"], suff_stat, n, sigma_DP, prior_chi, prior_nu,
                                                           max_ent_dist)

    fail_count = 0
    for i in range(max_retries + 1):
        lambdas = init_lambdas.clone().requires_grad_(True)
        success, lambdas, losses = laplace_optimisation(lambdas, potential_fn, max_iters, tol, max_loss_jump)
        if success:
            break
        else:
            fail_count += 1

    hess = autograd.functional.hessian(lambda l: potential_fn({"lambdas": l}), lambdas)
    laplace_approx = torch.distributions.MultivariateNormal(lambdas, precision_matrix=hess)

    rng_state_restore(ors)
    return laplace_approx, losses, fail_count


def tqdm_choice(iter: Iterable, choice):
    return tqdm.tqdm(iter) if choice else iter


def generate_synthetic_data(posterior_values, n_syn_dataset, max_ent_dist, show_progressbar=True):
    n_syn_datasets = posterior_values.shape[0]
    syn_datasets = np.zeros((n_syn_datasets, n_syn_dataset, max_ent_dist.d))
    for i in tqdm_choice(range(n_syn_datasets), show_progressbar):
        syn_datasets[i, :, :] = max_ent_dist.sample(torch.tensor(posterior_values[i, :]),
                                                    n_syn_dataset).detach().numpy()
    return syn_datasets


def estimate_query_mean(posterior_values, max_ent_dist, queries):
    n_syn_datasets = posterior_values.shape[0]
    return torch.stack(
        [max_ent_dist.mean_query_values(queries, posterior_values[i, :]) for i in range(n_syn_datasets)]).mean(dim=0)
