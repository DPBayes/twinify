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


import unittest

from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from twinify.napsu_mq.log_factor import LogFactor

class LogFactorJaxTest(unittest.TestCase):

    def test_log_factor_product_overlapping_scopes(self):
        a = jnp.arange(3 * 2).reshape((3, 2))
        b = jnp.arange(4).reshape((2, 2)) + 5
        af = LogFactor([0, 1], a)
        bf = LogFactor([1, 3], b)

        c = af.product(bf)
        self.assertSetEqual(set(c.scope), {0, 1, 3})
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    self.assertEqual(c.values[i, j, k], a[i, j] + b[j, k])

    def test_log_factor_product_same_scopes(self):
        a = jnp.arange(4).reshape((2, 2))
        b = jnp.arange(4).reshape((2, 2)) + 5
        af = LogFactor([0, 1], a)
        bf = LogFactor([0, 1], b)
        c = af.product(bf)

        self.assertSetEqual(set(c.scope), {0, 1})
        self.assertTrue(jnp.allclose(c.values, a + b))

    def test_log_factor_product_different_scopes(self):
        a = jnp.arange(4).reshape((2, 2))
        b = jnp.arange(4).reshape((2, 2)) + 5
        af = LogFactor([0, 1], a)
        bf = LogFactor([2, 4], b)
        c = af.product(bf)

        self.assertSetEqual(set(c.scope), {0, 1, 2, 4})
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        self.assertEqual(c.values[i, j, k, l], a[i, j] + b[k, l])

    def test_log_factor_product_subset_scope(self):
        a = jnp.arange(3 * 2 * 2).reshape((3, 2, 2))
        b = jnp.arange(3 * 2).reshape((3, 2))
        af = LogFactor([0, 1, 2], a)
        bf = LogFactor([0, 1], b)
        c = af.product(bf)

        self.assertSetEqual(set(c.scope), {0, 1, 2})
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    self.assertEqual(c.values[i, j, k], a[i, j, k] + b[i, j])

    def test_log_factor_marginalise(self):
        a = jnp.arange(3 * 2 * 2).reshape((3, 2, 2))
        af = LogFactor([0, 1, 4], a)

        m0 = af.marginalise(0)
        self.assertSetEqual(set(m0.scope), {1, 4})
        for i in range(2):
            for j in range(2):
                self.assertEqual(m0.values[i, j], jnp.log(
                    jnp.exp(af.values[0, i, j]) + jnp.exp(af.values[1, i, j]) + jnp.exp(af.values[2, i, j])))

        m1 = af.marginalise(1)
        self.assertSetEqual(set(m1.scope), {0, 4})
        for i in range(3):
            for j in range(2):
                self.assertEqual(m1.values[i, j], jnp.log(jnp.exp(af.values[i, 0, j]) + jnp.exp(af.values[i, 1, j])))

    def test_condition(self):
        a = jnp.arange(3 * 2 * 2).reshape((3, 2, 2))
        af = LogFactor([0, 1, 4], a)

        m0 = af.condition(0, 2)
        self.assertSetEqual(set(m0.scope), {1, 4})
        for i in range(2):
            for j in range(2):
                self.assertEqual(m0.values[i, j], af.values[2, i, j])

        m1 = af.condition(1, 1)
        self.assertSetEqual(set(m1.scope), {0, 4})
        for i in range(3):
            for j in range(2):
                self.assertEqual(m1.values[i, j], af.values[i, 1, j])

    def test_add_batch_dim(self):
        a = jnp.arange(3 * 2 * 2).reshape((3, 2, 2))
        af = LogFactor([0, 1, 4], a)
        b = af.add_batch_dim(10)
        self.assertTupleEqual(b.scope, ("batch",) + af.scope)
        self.assertTupleEqual(tuple(b.values.shape), (10,) + tuple(af.values.shape))
        for i in range(10):
            self.assertTrue(jnp.allclose(b.values[i], af.values))

    def test_batch_condition(self):
        a = jnp.arange(3 * 2 * 2).reshape((3, 2, 2))
        af = LogFactor([0, 1, 4], a)
        b = af.add_batch_dim(4)
        b.values = b.values.at[1].set(jnp.zeros((3, 2, 2)))
        b.values = b.values.at[2].set(jnp.ones((3, 2, 2)))

        m1 = b.batch_condition(0, jnp.array((0, 1, 1, 2)))
        self.assertTupleEqual(tuple(m1.values.shape), (4, 2, 2))
        self.assertTrue(jnp.allclose(m1.values[0], af.values[0, :, :]))
        self.assertTrue(jnp.allclose(m1.values[1], jnp.zeros((2, 2))))
        self.assertTrue(jnp.allclose(m1.values[2], jnp.ones((2, 2))))
        self.assertTrue(jnp.allclose(m1.values[3], af.values[2, :, :]))

        m2 = b.batch_condition(1, jnp.array((0, 1, 0, 1)))
        self.assertTupleEqual(tuple(m2.values.shape), (4, 3, 2))
        self.assertTrue(jnp.allclose(m2.values[0], af.values[:, 0, :]))
        self.assertTrue(jnp.allclose(m2.values[1], jnp.zeros((3, 2))))
        self.assertTrue(jnp.allclose(m2.values[2], jnp.ones((3, 2))))
        self.assertTrue(jnp.allclose(m2.values[3], af.values[:, 1, :]))


if __name__ == "__main__":
    unittest.main()
