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

import unittest
import pandas as pd
import jax
import jax.numpy as jnp
from jax.dtypes import canonicalize_dtype
import numpy as np

import numpyro.distributions as dists
import numbers

from twinify.dpvi.modelling.automodel import Distribution

class DistributionTest(unittest.TestCase):



    def test_get_support_dtype(self) -> None:
        self.assertEqual(Distribution.get_support_dtype(dists.Normal), canonicalize_dtype('float'))
        self.assertEqual(Distribution.get_support_dtype(dists.HalfNormal), canonicalize_dtype('float'))
        self.assertEqual(Distribution.get_support_dtype(dists.BernoulliProbs), canonicalize_dtype('int'))
        self.assertEqual(Distribution.get_support_dtype(dists.CategoricalProbs(np.ones((2,))*.5)), canonicalize_dtype('int'))
        self.assertEqual(Distribution.get_support_dtype(dists.Poisson), canonicalize_dtype('int'))

    def test_normal(self) -> None:
        d = Distribution.from_distribution_instance(dists.Normal())
        self.assertEqual('Normal', d.name)

        self.assertEqual(canonicalize_dtype('float'), d.support_dtype)
        self.assertFalse(d.has_data_dependent_shape)
        self.assertFalse(d.is_categorical)
        self.assertFalse(d.is_discrete)
        self.assertEqual(dists.Normal, d.numpyro_class)

        parameter_transforms = d.parameter_transforms
        for arg_name in ['loc', 'scale']:
            self.assertIsInstance(
                parameter_transforms[arg_name],
                type(dists.transforms.biject_to(dists.Normal.arg_constraints[arg_name]))
            )

        instance_default_params = d.instantiate()
        self.assertEqual(jnp.array([0.]), instance_default_params.mean)
        self.assertEqual(jnp.array([1.]), instance_default_params.scale)

        instance_with_params = d.instantiate({'loc': .1, 'scale': 3.})
        self.assertEqual(jnp.array([.1]), instance_with_params.mean)
        self.assertEqual(jnp.array([3.]), instance_with_params.scale)

    def test_bernoulli(self) -> None:
        d = Distribution.from_distribution_instance(dists.BernoulliProbs(.5))
        self.assertEqual('BernoulliProbs', d.name)

        self.assertEqual(canonicalize_dtype('int'), d.support_dtype)
        self.assertFalse(d.has_data_dependent_shape)
        self.assertTrue(d.is_categorical)
        self.assertTrue(d.is_discrete)
        self.assertEqual(dists.BernoulliProbs, d.numpyro_class)

        parameter_transforms = d.parameter_transforms
        for arg_name in ['probs']:
            self.assertIsInstance(
                parameter_transforms[arg_name],
                type(dists.transforms.biject_to(dists.BernoulliProbs.arg_constraints[arg_name]))
            )

        instance_default_params = d.instantiate()
        self.assertEqual(jnp.array([.5]), instance_default_params.mean)

        instance_with_params = d.instantiate({'probs': .1})
        self.assertEqual(jnp.array([.1]), instance_with_params.mean)

    def test_poisson(self) -> None:
        d = Distribution.from_distribution_instance(dists.Poisson(2.))
        self.assertEqual('Poisson', d.name)

        self.assertEqual(canonicalize_dtype('int'), d.support_dtype)
        self.assertFalse(d.has_data_dependent_shape)
        self.assertFalse(d.is_categorical)
        self.assertTrue(d.is_discrete)
        self.assertEqual(dists.Poisson, d.numpyro_class)

        parameter_transforms = d.parameter_transforms
        for arg_name in ['rate']:
            self.assertIsInstance(
                parameter_transforms[arg_name],
                type(dists.transforms.biject_to(dists.Poisson.arg_constraints[arg_name]))
            )

        instance_default_params = d.instantiate()
        self.assertEqual(jnp.array([1.]), instance_default_params.rate)

        instance_with_params = d.instantiate({'rate': 5.5})
        self.assertEqual(jnp.array([5.5]), instance_with_params.rate)

    def test_categorical(self) -> None:
        d = Distribution.from_distribution_instance(dists.CategoricalProbs(jnp.array([.1, .3, .6])))
        self.assertEqual('CategoricalProbs', d.name)

        self.assertEqual(canonicalize_dtype('int'), d.support_dtype)
        self.assertTrue(d.has_data_dependent_shape)
        self.assertTrue(d.is_categorical)
        self.assertTrue(d.is_discrete)
        self.assertEqual(dists.CategoricalProbs, d.numpyro_class)

        parameter_transforms = d.parameter_transforms
        for arg_name in ['probs']:
            self.assertIsInstance(
                parameter_transforms[arg_name],
                type(dists.transforms.biject_to(dists.CategoricalProbs.arg_constraints[arg_name]))
            )

        instance_default_params = d.instantiate()
        self.assertTrue(np.allclose(jnp.array([.5, .5]), instance_default_params.probs))

        instance_with_params = d.instantiate({'probs': jnp.array([.2, .5, .3])})
        self.assertTrue(np.allclose(jnp.array([.2, .5, .3]), instance_with_params.probs))

