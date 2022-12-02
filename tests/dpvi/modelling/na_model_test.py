import unittest
import jax
import numpy as np

import numpyro.distributions as dists

from twinify.dpvi.modelling.na_model import na_constraint, NAModel

class NAConstraintTest(unittest.TestCase):

    def test_constraint_eval(self) -> None:
        base_constraint = dists.constraints.positive
        constraint = na_constraint(base_constraint)

        self.assertTrue(constraint(3.))
        self.assertTrue(constraint(np.nan))
        self.assertFalse(constraint(-1.))

    def test_is_discrete(self) -> None:
        real_base_constraint = dists.constraints.positive
        real_constraint = na_constraint(real_base_constraint)

        self.assertFalse(real_constraint.is_discrete)

        discrete_base_constraint = dists.constraints.positive_integer
        discrete_constraint = na_constraint(discrete_base_constraint)

        self.assertTrue(discrete_constraint.is_discrete)

    def test_event_dim(self) -> None:
        vec_base_constraint = dists.constraints.real_vector
        vec_constraint = na_constraint(vec_base_constraint)

        self.assertEqual(vec_base_constraint.event_dim, vec_constraint.event_dim)

    def test_feasible_like(self) -> None:
        base_constraint = dists.constraints.positive
        constraint = na_constraint(base_constraint)

        prototype = np.array([[1., 2., 3.], [3., 4., 5.]])

        base_result = base_constraint.feasible_like(prototype)
        result = constraint.feasible_like(prototype)
        self.assertTrue(np.allclose(base_result, result))


class NAModelTest(unittest.TestCase):

    def test_init(self) -> None:
        base_dist = dists.Dirichlet(np.ones((3, 5,)))
        na_model = NAModel(base_dist, 0.2)
        self.assertEqual(base_dist.batch_shape, na_model.batch_shape)
        self.assertEqual(base_dist.event_shape, na_model.event_shape)
        self.assertIs(base_dist, na_model.base_distribution)
        self.assertIsInstance(na_model.support, na_constraint)
        self.assertEqual(base_dist.support, na_model.support.base_constraint)

    def test_enumerate_support(self) -> None:
        not_base_dist = dists.Dirichlet(np.ones((3, 5,)))
        assert not not_base_dist.has_enumerate_support
        not_na_model = NAModel(not_base_dist, 0.2)

        self.assertFalse(not_na_model.has_enumerate_support)

        yea_base_dist = dists.Binomial(5, probs=.2)
        assert yea_base_dist.has_enumerate_support
        yea_na_model = NAModel(yea_base_dist, 0.2)

        self.assertTrue(yea_na_model.has_enumerate_support)
        self.assertTrue(np.all(
            yea_base_dist.enumerate_support() == yea_na_model.enumerate_support()
        ))

    def test_sample(self) -> None:
        base_dist = dists.Dirichlet(np.ones((3, 5,)))
        na_rate = 0.2
        na_model = NAModel(base_dist, na_rate)

        key = jax.random.PRNGKey(924)
        sample_shape = (1000, 2)
        sample = na_model.sample(key, sample_shape)
        self.assertEqual((1000, 2, 3, 5), sample.shape)

        _, base_key = jax.random.split(key)
        base_sample = base_dist.sample(base_key, sample_shape)
        self.assertTrue(np.all(
            np.where(np.isnan(sample), True, np.isclose(base_sample, sample))
        ))

        self.assertTrue(
            np.isclose(na_rate, np.isnan(sample).mean(), atol=1e-1)
        )

    def test_sample_no_sample_shape(self) -> None:
        base_dist = dists.Dirichlet(np.ones((3, 5,)))
        na_rate = 0.2
        na_model = NAModel(base_dist, na_rate)

        key = jax.random.PRNGKey(924)
        sample = na_model.sample(key)
        self.assertEqual((3, 5), sample.shape)

        _, base_key = jax.random.split(key)
        base_sample = base_dist.sample(base_key)
        self.assertTrue(np.all(
            np.where(np.isnan(sample), True, np.isclose(base_sample, sample))
        ))

    def test_log_prob(self) -> None:
        base_dist = dists.Dirichlet(np.ones((3, 2,)))
        na_rate = 0.2
        na_model = NAModel(base_dist, na_rate)

        x = np.array([
            [
                [.2, .8], [np.nan, np.nan], [.1, .9]
            ],
            np.zeros((3, 2)) + np.nan
        ])
        nan_mask = np.array([[False, True, False], [True, True, True]])

        base_log_prob = base_dist.log_prob(x)
        base_log_prob = np.where(np.isnan(base_log_prob), 0., base_log_prob)
        nan_log_prob = dists.Bernoulli(probs = na_rate).log_prob(nan_mask)

        expected_log_prob = nan_log_prob + base_log_prob

        log_prob = na_model.log_prob(x)

        self.assertTrue(np.allclose(expected_log_prob, log_prob))

    def test_log_prob_no_event_shape(self) -> None:
        base_dist = dists.Normal(np.ones((3, 2,)))
        na_rate = 0.2
        na_model = NAModel(base_dist, na_rate)

        x = np.array([
            [np.nan, .8], [.2, np.nan], [.1, .9]
        ])
        nan_mask = np.array([[True, False], [False, True], [False, False]])

        base_log_prob = base_dist.log_prob(x)
        base_log_prob = np.where(np.isnan(base_log_prob), 0., base_log_prob)
        nan_log_prob = dists.Bernoulli(probs = na_rate).log_prob(nan_mask)

        expected_log_prob = nan_log_prob + base_log_prob

        log_prob = na_model.log_prob(x)

        self.assertTrue(np.allclose(expected_log_prob, log_prob))
