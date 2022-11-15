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
Missing value model used by twinify main script and available for modelling.
"""

import jax.numpy as jnp
import numpy as np
import jax
import typing

import numpyro.distributions as dists


class _NAConstraint(dists.constraints.Constraint):
    """ Wraps a constraint to additionally allow NaN values.
    """

    def __init__(self, base_constraint: dists.constraints.Constraint) -> None:
        self.base_constraint = base_constraint
        super().__init__()

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.isnan(value) | self.base_constraint(value)

    def feasible_like(self, prototype: jnp.ndarray) -> jnp.ndarray:
        return self.base_constraint.feasible_like(prototype)

    @property
    def is_discrete(self) -> bool:
        return self.base_constraint.is_discrete

    @property
    def event_dim(self) -> int:
        return self.base_constraint.event_dim


na_constraint = _NAConstraint


class NAModel(dists.Distribution):
    """ Model decorator for missing values.

    Can be used to enhance any single feature distribution to handle missing
    values (represented by nan values (`np.nan`)) in a missing completely at random fashion.
    """

    arg_constraints = {'na_prob': dists.constraints.unit_interval}

    def __init__(self, base_dist, na_prob=0.5, validate_args=None) -> None:
        """
        Initialize a new NAModel instance.

        Args:
            base_dist (numpyro.distributions.Distribution): The feature distribution for existing data.
            na_prob (float): Probability that data is missing.
        """
        self._base_dist = base_dist
        self._na_prob = na_prob
        super(NAModel, self).__init__(base_dist.batch_shape, base_dist.event_shape, validate_args=validate_args)

    @property
    def base_distribution(self) -> dists.Distribution:
        """ The feature base distribution for existing data. """
        return self._base_dist

    @property
    def support(self) -> dists.constraints.Constraint:
        return na_constraint(self._base_dist.support)

    @property
    def has_enumerate_support(self) -> bool:
        return self._base_dist.has_enumerate_support

    def enumerate_support(self, expand=True):
        return self._base_dist.enumerate_support(expand)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates the log-probabilities for given data realizations (observed or missing)

        Args:
            values (array_like): The data.
        Returns:
            array_like containing probabilities for each given data point
        """
        if len(self.event_shape) > 0:
            reshaped_value = value.reshape(*(value.shape[:-len(self.event_shape)]), -1)
            nans = jnp.isnan(reshaped_value).sum(-1) == np.prod(self.event_shape)
        else:
            nans = jnp.isnan(value)
        nan_log_probs = dists.Bernoulli(probs = self._na_prob).log_prob(nans)

        value_log_probs = self._base_dist.log_prob(value)
        return nan_log_probs + jnp.where(nans, 0., value_log_probs)

    def sample(self, key :jax.random.KeyArray, sample_shape: typing.Tuple[int] = ()) -> jnp.ndarray:
        """
        Samples from the missing values model.

        Args:
            key (jax.random.PRNGKey): RNG key.
            sample_shape (tuple): How many samples to draw and how to arrange them
        Returns:
            array_like
        """
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(
            self,
            key :jax.random.KeyArray,
            sample_shape: typing.Tuple[int] = ()
        ) -> typing.Tuple[jnp.ndarray, typing.Tuple[typing.Any]]:

        nan_rng, values_rng = jax.random.split(key)
        nan_mask = dists.Bernoulli(probs = self._na_prob).sample(
            nan_rng, sample_shape=(sample_shape + self.batch_shape)
        )

        values = self.base_distribution.sample(values_rng, sample_shape=sample_shape)

        # expand trailing dims for nan_mask to match event_shape
        nan_mask = jnp.expand_dims(nan_mask, range(len(nan_mask.shape) - len(values.shape), 0))

        nans = jnp.where(nan_mask, jnp.nan, 0.)

        return values + nans, ()

    def __repr__(self) -> str:
        return f"NAModel({repr(self.base_distribution)}"
