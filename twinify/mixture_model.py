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
Mixture model used by twinify main script and available for modelling.
"""

import numpy as np
import jax.numpy as jnp
import jax

import numpyro
import numpyro.distributions as dist

from jax.scipy.special import logsumexp
from numpyro.distributions.constraints import Constraint
import typing


class _CombinedConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self, base_constraints: typing.List[Constraint]) -> None:
        assert np.all([isinstance(base_constraint, Constraint) for base_constraint in base_constraints])

        self.base_constraints = base_constraints
        super().__init__()

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        if jnp.shape(value)[-1] != len(self.base_constraints):
            raise ValueError(f"Size of last dimension of value ({jnp.shape(value)[-1]})"\
                f"does match the number of constraints ({len(self.base_constraints)})!")

        results = [None] * len(self.base_constraints)
        for i, constraint in enumerate(self.base_constraints):
            results[i] = constraint(value[..., i])
        results = jnp.all(jnp.stack(results, axis=-1), axis=-1)
        return results

    def feasible_like(self, prototype: jnp.ndarray) -> jnp.ndarray:
        if jnp.shape(prototype)[-1] != len(self.base_constraints):
            raise ValueError(f"Size of last dimension of prototype ({jnp.shape(prototype)[-1]})"\
                f"does match the number of constraints ({len(self.base_constraints)})!")

        results = [None] * len(self.base_constraints)
        for i, constraint in enumerate(self.base_constraints):
            results[i] = constraint.feasible_like(prototype[..., i])

        results = jnp.stack(results, axis=-1)
        return results

combined_constraint = _CombinedConstraint


class MixtureModel(dist.Distribution):
    """ A general purpose mixture model """

    arg_constraints = {
        '_pis' : dist.constraints.simplex
    }

    @property
    def support(self) -> Constraint:
        return self._constraint

    def __init__(self, dists, pis=1.0, validate_args=None):
        """
        Initializes a MixtureModel instance.

        The feature distributions provided by argument `dists` must be in the
        same order as the features are provided in the `value` vector passed
        to `log_prob`.

        Args:
            dists (list of numpyro.distributions.Distribution): The list of feature distributions.
            pis (array_like): The mixture weights.
        """
        self.dists = dists
        event_shape = None
        batch_shape = None
        for dist in dists:
            if not isinstance(dist, numpyro.distributions.Distribution):
                raise ValueError("MixtureModel got an argument that is not a distribution.")
            if event_shape is None:
                event_shape = dist.event_shape
            else:
                if event_shape != dist.event_shape:
                    raise ValueError(f"Feature distribution event shape deviates from mixture event shape {event_shape}; got {dist.event_shape} from {dist}.")
            if batch_shape is None:
                batch_shape = dist.batch_shape
            else:
                if batch_shape != dist.batch_shape:
                    raise ValueError(f"Feature distribution batch shape deviates from mixture batch shape {batch_shape}; got {dist.batch_shape} from {dist}.")
        self._mixture_components = batch_shape[-1]
        batch_shape = batch_shape[:-1] if len(batch_shape) > 0 else 1
        event_shape = (len(dists),) + event_shape
        if len(pis) != self._mixture_components:
            raise ValueError(f"Mixture model has {self._mixture_components} components but only got {len(pis)} mixture weights.")
        self._pis = pis

        constraints = [dist.support for dist in self.dists]
        self._constraint = combined_constraint(constraints)

        super(MixtureModel, self).__init__(batch_shape, event_shape, validate_args)

    @property
    def mixture_components(self):
        return self._mixture_components

    def log_prob(self, value):
        """
        Evaluates the log-probabilities for given observations.

        The log-probabilities of the K component mixture model
        $\log p(x | \theta)  =  \log \sum_{k=1}^K \pi_k p(x | \theta_k)$
        where $p(x | \theta_k)$ denotes the probabilities of the k-th mixture cluster

        Args:
            values (array_like): The observations. Dimension of the last axis must
                be the number of features (i.e., `len(dists)`).
        Returns:
            array_like of shape `values.shape[:-1]`
        """
        log_phis = jnp.array([
                dbn.log_prob(value[..., feat_idx, jnp.newaxis])
                for feat_idx, dbn in enumerate(self.dists)
            ]).sum(axis=0)
        log_pis = jnp.log(self._pis)

        temp = log_pis + log_phis

        return logsumexp(temp, axis=-1)

    def sample(self, key, sample_shape=()):
        """
        Samples from the mixture model.

        Args:
            key (jax.random.PRNGKey): RNG key.
            sample_shape (tuple): How many samples to draw and how to arrange them
        Returns:
            array_like of shape `(*sample_shape, len(dists))`
        """
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        num_samples = int(np.prod(sample_shape))

        keys = jax.random.split(key, num_samples)

        @jax.vmap
        def sample_single(single_key):
            vals_rng_key, pis_rng_key = jax.random.split(single_key, 2)
            z = dist.Categorical(self._pis).sample(pis_rng_key)
            rng_keys = jax.random.split(vals_rng_key, len(self.dists))
            vals = [dbn.sample(rng_keys[feat_idx])[z] \
                    for feat_idx, dbn in enumerate(self.dists)]
            return jnp.stack(vals).T, z

        vals, zs = sample_single(keys)
        if len(sample_shape) == 0:
            vals = vals.squeeze(0)

        zs = jnp.reshape(zs, sample_shape)
        final_vals_shape = sample_shape + jnp.shape(jnp.atleast_2d(vals))[1:]
        vals = jnp.reshape(vals, final_vals_shape)
        return vals, [zs]
