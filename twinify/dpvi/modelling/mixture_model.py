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

"""
Mixture model used by twinify main script and available for modelling.
"""

import numpy as np
import jax.numpy as jnp
import jax

import numpyro
import numpyro.distributions as dists

from jax.scipy.special import logsumexp
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import biject_to
import typing


class _CombinedConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.
    """

    def __init__(self,
            base_constraints: typing.Iterable[Constraint],
            sizes: typing.Optional[typing.Iterable[int]] = None) -> None:

        self._base_constraints = list()

        event_dims = 1
        for i, base_constraint in enumerate(base_constraints):
            if not isinstance(base_constraint, Constraint):
                raise ValueError(f"Element {i} in constraint list is not of type Constraint.")

            if base_constraint.event_dim > 1:
                raise ValueError(
                    f"Currently only support base constraints with event_dim <= 1 but element {i}"
                    f"in constraint list has event_dim = {base_constraint.event_dim}"
                )

            self._base_constraints.append(base_constraint)

        self._event_dims = event_dims
        if sizes is not None:
            sizes = np.array(sizes)
            if len(sizes) != len(self._base_constraints):
                raise ValueError(
                    f"Number of sizes {len(sizes)} has to equal number of base constraints {(len(self._base_constraints))}."
                )

            self._dim_offsets = tuple(np.cumsum(np.hstack((0, sizes))))
        else:
            self._dim_offsets = tuple(range(len(self._base_constraints) + 1))

        assert len(self._dim_offsets) == len(self._base_constraints) + 1

        super().__init__()

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        if jnp.shape(value)[-1] != self.size:
            raise ValueError(f"Last dimension of value ({jnp.shape(value)[-1]})"\
                f"does match the constraint size ({self.size})!")

        base_results = [
            jnp.all(constraint(value[..., start:end]))
            for constraint, start, end in zip(self._base_constraints, self._dim_offsets[:-1], self._dim_offsets[1:])
        ]
        return jnp.all(jnp.stack(base_results, -1))

    def feasible_like(self, prototype: jnp.ndarray) -> jnp.ndarray:
        if jnp.shape(prototype)[-1] != self.size:
            raise ValueError(f"Last dimension of value ({jnp.shape(prototype)[-1]})"\
                f"does match the constraint size ({self.size})!")

        base_results = [
            constraint.feasible_like(prototype[..., start:end]).reshape(jnp.shape(prototype)[:-1] + (end - start,))
            for constraint, start, end in zip(self._base_constraints, self._dim_offsets[:-1], self._dim_offsets[1:])
        ]

        results = jnp.concatenate(base_results, axis=-1)
        return results

    @property
    def is_discrete(self) -> bool:
        return np.all(tuple(constraint.is_discrete for constraint in self._base_constraints))

    @property
    def event_dim(self) -> int:
        return self._event_dims

    @property
    def size(self) -> int:
        return self._dim_offsets[-1]

    @property
    def offsets(self) -> typing.Iterable[int]:
        return self._dim_offsets[:-1]
    
    def tree_flatten(self) -> typing.Tuple[typing.Iterable[Constraint], typing.Dict[str, typing.Any]]:
        return self._base_constraints, {'sizes': np.diff(self._dim_offsets)}
    
    @classmethod
    def tree_unflatten(cls, aux_data: typing.Dict[str, typing.Any], params: typing.Iterable[Constraint]) -> "_CombinedConstraint":
        sizes = aux_data['sizes']
        return _CombinedConstraint(
            params, sizes
        )
    

combined_constraint = _CombinedConstraint
# TODO: need to implement a transform for this

class MixtureModel(dists.Distribution):
    """ A general purpose mixture model.

    Described a number of distributions that independently model feature dimensions
    of the data.
    The MixtureModel then is a mixture of components which all follow the
    structure described by the joint feature distributions.

    Feature distributions therefore must have a leading batch dimension that corresponds
    to the number of mixture components.

    Note this class behaves quite differently from `numpyro.distributions.Mixture`,
    where each distribution describes all features for a single mixture component.
     """

    arg_constraints = {
        'pis' : dists.constraints.simplex
    }

    has_enumerate_support = False
    reparametrized_params = ["pis"]

    @property
    def support(self) -> Constraint:
        return self._constraint

    @property
    def pis(self) -> jnp.ndarray:
        return self._pis

    def __init__(self, distributions: typing.Iterable[dists.Distribution], pis: np.ndarray, *, validate_args=None):
        """
        Initializes a MixtureModel instance.

        The feature distributions provided by argument `dists` must be in the
        same order as the features are provided in the `value` vector passed
        to `log_prob`.

        Args:
            distributions (list of numpyro.distributions.Distribution): The list of feature distributions.
            pis (array_like): The mixture weights.
        """
        self.distributions = list(distributions)
        batch_shape = None
        constraints = []
        sizes = []

        self._pis = pis
        self._mixture_components = len(pis)

        for dist in distributions:
            if not isinstance(dist, dists.Distribution):
                raise ValueError("MixtureModel got an argument that is not a distribution.")
            if len(dist.event_shape) > 1:
                raise ValueError(f"Feature distribution event shape cannot have more than one dimensions, was {dist.event_shape}.")

            if batch_shape is None:
                batch_shape = dist.batch_shape
                if len(batch_shape) == 0 or batch_shape[-1] != self._mixture_components:
                    raise ValueError(
                        f"Last batch dimension must equal the number of mixture components ({self._mixture_components})"
                        f" for all distributions; got {dist.batch_shape} from {dist}.")
            else:
                if batch_shape != dist.batch_shape:
                    raise ValueError(f"Feature distribution batch shape deviates from mixture batch shape {batch_shape}; got {dist.batch_shape} from {dist}.")

            sizes.append(dist.event_shape[-1] if len(dist.event_shape) > 0 else 1)
            constraints.append(dist.support)

        assert len(batch_shape) > 0
        batch_shape = batch_shape[:-1]
        self._offsets = np.cumsum(np.hstack((0., sizes)), dtype=np.int32)
        event_shape = (self._offsets[-1],)

        if len(pis) != self._mixture_components:
            raise ValueError(f"Mixture model has {self._mixture_components} components but only got {len(pis)} mixture weights.")

        self._constraint = combined_constraint(constraints, sizes)

        super(MixtureModel, self).__init__(batch_shape, event_shape, validate_args=validate_args)

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
        log_phis = [
            dbn.log_prob(value[..., np.newaxis, start:stop])
            if len(dbn.event_shape) > 0
            else dbn.log_prob(value[..., np.newaxis, start])
            for dbn, start, stop in zip(self.distributions, self._offsets[:-1], self._offsets[1:])
        ]
        log_phis = jnp.array(log_phis).sum(0)
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
            array_like of shape `sample_shape + self.batch_shape + self.event_shape`
        """
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        batch_shape = sample_shape + self.batch_shape

        vals_rng_key, pis_rng_key = jax.random.split(key, 2)
        zs = dists.Categorical(self._pis).sample(pis_rng_key, sample_shape=batch_shape).flatten()

        component_keys = jax.random.split(vals_rng_key, len(self.distributions))
        vals = []
        for dbn, dbn_key in zip(self.distributions, component_keys):
            dbn_sample = dbn.sample(dbn_key, sample_shape=sample_shape)
            dbn_event_shape = dbn.event_shape
            if dbn_event_shape == ():
                dbn_event_shape = (1,)

            flat_batch_size = int(np.prod(batch_shape))

            dbn_sample = dbn_sample.reshape(flat_batch_size, self._mixture_components, *dbn_event_shape)
            dbn_sample = dbn_sample[np.arange(flat_batch_size), zs]
            dbn_sample = dbn_sample.reshape(*batch_shape, *dbn_event_shape)
            vals.append(dbn_sample)

        return jnp.concatenate(vals, axis=-1), [zs.reshape(batch_shape)]
