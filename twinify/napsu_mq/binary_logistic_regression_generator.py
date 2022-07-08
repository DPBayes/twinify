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
from typing import Optional

import torch
import itertools


class BinaryLogisticRegressionDataGenerator:
    """
    Generator for d-dimensional binary datasets where all variables but the
    last are generated with independent coin flips, and the last variable
    is generated from logistic regression on the others.

    For testing purposes.
    """

    def __init__(self, true_params: torch.Tensor):
        """Initialise the generator.
        Args:
            true_params (torch.tensor): Coefficients for the logistic regression.
        """
        self.true_params = true_params
        self.d = true_params.shape[0] + 1
        self.x_values = self.compute_x_values()
        self.values_by_feature = {i: [0, 1] for i in range(self.d)}

    def generate_data(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate the output d-dimensional binary output.
        Args:
            n (int): Number of datapoints to generate.
            generator (Pytorch RNG, optional): Random number generator. Defaults to None.
        Returns:
            torch.tensor: The generated output.
        """
        x = torch.bernoulli(torch.full((n, self.d - 1), 0.5), generator=generator)
        alpha = x @ self.true_params.view((-1, 1))
        probs = torch.special.expit(alpha)
        y = torch.bernoulli(probs, generator=generator)
        return torch.concat((x, y), dim=1)

    def compute_x_values(self) -> torch.Tensor:
        """Enumerate all possible datapoints.
        Returns:
            torch.tensor: 2-d array enumerating all possible datapoints.
        """
        x_values = torch.zeros((2 ** self.d, self.d))
        for i, val in enumerate(itertools.product(range(2), repeat=self.d)):
            x_values[i, :] = torch.tensor(val)
        return x_values
