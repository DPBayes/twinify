import unittest
import jax
import numpy as np

import numpyro.distributions as dists
from scipy.special import logsumexp

from twinify.dpvi.modelling.mixture_model import MixtureModel, combined_constraint

class CombinedConstraintTest(unittest.TestCase):

    def test_constraint_eval(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]
        constraint = combined_constraint(base_constraints)

        self.assertEqual(1, constraint.event_dim)
        self.assertEqual((0, 1, 2), constraint.offsets)
        self.assertEqual(3, constraint.size)

        self.assertTrue(constraint(np.array([3.1, 1, 2.])))
        self.assertFalse(constraint(np.array([-1., 1, 2.])))
        self.assertFalse(constraint(np.array([3.1, 1.5, 2.])))

        with self.assertRaises(ValueError):
            constraint(np.array([3.1, 1, 2., 4.]))
            constraint(np.array([3.1, 1]))

    def test_constraint_rejects_bad_sizes(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]

        with self.assertRaises(ValueError):
            combined_constraint(base_constraints, [1, 1])  # too short
            combined_constraint(base_constraints, [1, 1, 2, 3])  # too long

    def test_constraint_eval_with_sizes(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]
        sizes = [1, 2, 3]
        constraint = combined_constraint(base_constraints, sizes)

        self.assertEqual(1, constraint.event_dim)
        self.assertEqual((0, 1, 3), constraint.offsets)
        self.assertEqual(6, constraint.size)

        self.assertTrue(constraint(np.array([3.1, 1, 2, -1.1, 2.2, 0.1])))
        self.assertFalse(constraint(np.array([-3.1, 1, 2, -1.1, 2.2, 0.1])))
        self.assertFalse(constraint(np.array([3.1, 1, 2.1, -1.1, 2.2, 0.1])))
        self.assertFalse(constraint(np.array([3.1, -1, 2, -1.1, 2.2, 0.1])))

        with self.assertRaises(ValueError):
            constraint(np.array([3.1, 1, 2., 4.]))

    def test_is_discrete(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]
        constraint = combined_constraint(base_constraints)

        self.assertFalse(constraint.is_discrete)


        self.assertTrue(constraint(np.array([3.1, 1, 2.])))
        self.assertFalse(constraint(np.array([-1., 1, 2.])))
        self.assertFalse(constraint(np.array([3.1, 1.5, 2.])))

        with self.assertRaises(ValueError):
            constraint(np.array([3.1, 1, 2., 4.]))
            constraint(np.array([3.1, 1]))

    def test_feasible_like(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]
        constraint = combined_constraint(base_constraints)

        prototype = np.array([[1., 2., 3.], [3., 4., 5.]])
        expected = np.stack([
                base_constraint.feasible_like(prototype[:, i])
                for i, base_constraint in enumerate(base_constraints)
            ],
            axis=-1
        )

        result = constraint.feasible_like(prototype)
        self.assertTrue(np.allclose(expected, result))

        with self.assertRaises(ValueError):
            constraint.feasible_like([1., 2.])  # too short
            constraint.feasible_like([1., 2., 3., 4.])  # too long

    def test_feasible_like_with_sizes(self) -> None:
        base_constraints = [dists.constraints.positive, dists.constraints.positive_integer, dists.constraints.real_vector]
        sizes = [1, 2, 3]
        constraint = combined_constraint(base_constraints, sizes)

        prototype = np.array([[1., 2., 3., 4., 5., 6.], [3., 4., 5., 10., 11., 12.]])
        expected = np.concatenate([
                base_constraints[0].feasible_like(prototype[:, 0:1]),
                base_constraints[1].feasible_like(prototype[:, 1:3]),
                base_constraints[2].feasible_like(prototype[:, 3:6]),
            ],
            axis=-1
        )

        result = constraint.feasible_like(prototype)
        self.assertTrue(np.allclose(expected, result))

        with self.assertRaises(ValueError):
            constraint.feasible_like([1., 2., 3.])


class MixtureModelTest(unittest.TestCase):

    def test_init_fails_for_mixture_component_mismatch(self) -> None:
        base_dists = [dists.Normal(np.zeros((4, 8, 2)), 1.).to_event(1), dists.Poisson(np.ones((4, 8)))]
        pis = np.ones((2,)) / 2
        with self.assertRaises(ValueError):
            MixtureModel(base_dists, pis)

    def test_init(self) -> None:
        base_dists = [dists.Normal(np.zeros((4, 8, 2)), 1.).to_event(1), dists.Poisson(np.ones((4, 8)))]
        pis = np.ones((8,)) / 8
        mixture_model = MixtureModel(base_dists, pis)

        self.assertTrue(1, mixture_model.event_dim)
        self.assertTrue((3,), mixture_model.event_shape)
        self.assertTrue((4,), mixture_model.batch_shape)

    def test_log_prob(self) -> None:
        poisson_rate = np.tile(np.arange(8, dtype=np.float32) + 1, 4).reshape(4, 8)
        base_dists = [dists.Normal(np.zeros((4, 8, 2)), 1.).to_event(1), dists.Poisson(poisson_rate)]
        pis = np.ones((8,)) / 8
        mixture_model = MixtureModel(base_dists, pis)

        data = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
        data_first_part = data[:, :, np.newaxis, :2]
        data_second_part = data[:, :, np.newaxis, 2]

        expected = base_dists[0].log_prob(data_first_part) + base_dists[1].log_prob(data_second_part)
        expected = expected + np.log(pis)
        expected = logsumexp(expected, axis=-1)
        assert expected.shape == (5, 4)

        result = mixture_model.log_prob(data)
        self.assertTrue(np.allclose(expected, result))

    def test_sample(self) -> None:
        poisson_rate = np.tile(np.arange(8, dtype=np.float32) + 1, 4).reshape(4, 8)
        base_dists = [dists.Normal(np.zeros((4, 8, 2)), 1.).to_event(1), dists.Poisson(poisson_rate)]
        pis = np.ones((8,)) / 8
        mixture_model = MixtureModel(base_dists, pis)

        rng_key = jax.random.PRNGKey(0)
        sample, aux = mixture_model.sample_with_intermediates(rng_key, (10, 2))
        zs = aux[0]
        self.assertEqual((10, 2, 4, 3), sample.shape)
        self.assertEqual((10, 2, 4), zs.shape)

        vals_rng_key, _ = jax.random.split(rng_key, 2)
        dbn_keys = jax.random.split(vals_rng_key)

        x_first = base_dists[0].sample(dbn_keys[0], sample_shape=(10, 2))
        x_second = base_dists[1].sample(dbn_keys[1], sample_shape=(10, 2))

        for i in range(10):
            for j in range(2):
                for k in range(4):
                    self.assertTrue(np.allclose(x_first[i, j, k, zs[i, j, k]], sample[i, j, k, :2]))
                    self.assertTrue(np.allclose(x_second[i, j, k, zs[i, j, k]], sample[i, j, k, 2]))


    def test_support(self) -> None:
        poisson_rate = np.tile(np.arange(8, dtype=np.float32) + 1, 4).reshape(4, 8)
        base_dists = [dists.Normal(np.zeros((4, 8, 2)), 1.).to_event(1), dists.Poisson(poisson_rate)]
        pis = np.ones((8,)) / 8
        mixture_model = MixtureModel(base_dists, pis)

        expected = combined_constraint(
            [dists.constraints.independent(dists.constraints.real, 1), dists.constraints.positive_integer],
            [2, 1]
        )

        self.assertIsInstance(mixture_model.support, combined_constraint)
        self.assertEqual(expected.offsets, mixture_model.support.offsets)
        self.assertEqual(type(expected._base_constraints[0]), type(mixture_model.support._base_constraints[0]))
        self.assertEqual(type(expected._base_constraints[1]), type(mixture_model.support._base_constraints[1]))
