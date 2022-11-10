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

import numpy as np

import twinify.napsu_mq.privacy_accounting as privacy_accounting
from scipy import special


def phi(t):
    return (1 + special.erf(t / np.sqrt(2))) / 2


def calculate_delta_bound(sensitivity, sigma, epsilon):
    return phi((sensitivity / (2 * sigma)) - ((epsilon * sigma) / sensitivity)) - (np.e ** epsilon) * phi(
        -(sensitivity / (2 * sigma)) - ((epsilon * sigma) / sensitivity))


def calculate_epsilon_from_rho(rho, delta):
    epsilon = rho + 2 * np.sqrt(rho * np.log(1/delta))
    return epsilon


def calculate_sensitivity_from_rho(rho, sigma):
    sensitivity = sigma ** 2 * (2 * rho)
    return sensitivity


class PrivacyAccountingTest(unittest.TestCase):

    def test_sigma(self):
        with self.assertRaises(ZeroDivisionError):
            privacy_accounting.sigma(epsilon=0, delta_bound=0, sensitivity=0)

        delta_bound = calculate_delta_bound(sensitivity=np.sqrt(2), sigma=1, epsilon=np.sqrt(2))
        sigma = privacy_accounting.sigma(epsilon=np.sqrt(2), delta_bound=delta_bound, sensitivity=np.sqrt(2))
        self.assertAlmostEqual(sigma, 1)

        delta_bound = calculate_delta_bound(sensitivity=2, sigma=2, epsilon=1)
        sigma = privacy_accounting.sigma(epsilon=1, delta_bound=delta_bound, sensitivity=2)
        self.assertAlmostEqual(sigma, 2)

        delta_bound = calculate_delta_bound(sensitivity=10, sigma=5, epsilon=8)
        sigma = privacy_accounting.sigma(epsilon=8, delta_bound=delta_bound, sensitivity=10)
        self.assertAlmostEqual(sigma, 5)

        delta_bound = calculate_delta_bound(sensitivity=2, sigma=10, epsilon=0.1)
        sigma = privacy_accounting.sigma(epsilon=0.1, delta_bound=delta_bound, sensitivity=2)
        self.assertAlmostEqual(sigma, 10)

    def test_delta(self):
        delta = privacy_accounting.delta(epsilon=1, sens_per_sigma=1)
        delta_bound = calculate_delta_bound(sensitivity=1, sigma=1, epsilon=1)
        self.assertAlmostEqual(delta, delta_bound)

        delta = privacy_accounting.delta(epsilon=10, sens_per_sigma=2)
        delta_bound = calculate_delta_bound(sensitivity=4, sigma=2, epsilon=10)
        self.assertAlmostEqual(delta, delta_bound)

        delta = privacy_accounting.delta(epsilon=0.1, sens_per_sigma=5)
        delta_bound = calculate_delta_bound(sensitivity=10, sigma=2, epsilon=0.1)
        self.assertAlmostEqual(delta, delta_bound)

    def test_find_sens_per_sigma(self):
        delta_bound = calculate_delta_bound(epsilon=1, sensitivity=2, sigma=1)
        sens_per_sigma = privacy_accounting.find_sens_per_sigma(epsilon=1, delta_bound=delta_bound)
        self.assertAlmostEqual(sens_per_sigma, 2)

        delta_bound = calculate_delta_bound(epsilon=0.1, sensitivity=10, sigma=3)
        sens_per_sigma = privacy_accounting.find_sens_per_sigma(epsilon=0.1, delta_bound=delta_bound)
        self.assertAlmostEqual(sens_per_sigma, 10 / 3)

        delta_bound = calculate_delta_bound(epsilon=10, sensitivity=4, sigma=1)
        sens_per_sigma = privacy_accounting.find_sens_per_sigma(epsilon=10, delta_bound=delta_bound)
        self.assertAlmostEqual(sens_per_sigma, 4)

    def test_eps_delta_budget_to_rho_budget(self):
        rho = privacy_accounting.eps_delta_budget_to_rho_budget(epsilon=0.5, delta=0.1)
        epsilon_derived_from_rho = calculate_epsilon_from_rho(rho=rho, delta=0.1)
        self.assertAlmostEqual(epsilon_derived_from_rho, 0.5)

        rho = privacy_accounting.eps_delta_budget_to_rho_budget(epsilon=1, delta=0.01)
        epsilon_derived_from_rho = calculate_epsilon_from_rho(rho=rho, delta=0.01)
        self.assertAlmostEqual(epsilon_derived_from_rho, 1)

        rho = privacy_accounting.eps_delta_budget_to_rho_budget(epsilon=10, delta=0.01)
        epsilon_derived_from_rho = calculate_epsilon_from_rho(rho=rho, delta=0.01)
        self.assertAlmostEqual(epsilon_derived_from_rho, 10)