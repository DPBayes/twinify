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

import pandas as pd

from typing import Optional, Callable, Iterable

import numpy as np
import jax
import jax.numpy as jnp

import d3p.minibatch
import d3p.random
import d3p.dputil
import d3p.svi
import numpyro.infer
from twinify.base import InferenceModel, InferenceResult

from twinify.dpvi import PrivacyLevel, ModelFunction, GuideFunction
from twinify.dpvi.dpvi_result import DPVIResult


class InferenceException(Exception):

    def __init__(self, iteration: int, total_iterations: int) -> None:
        self._iteration = iteration
        self._total_iterations = total_iterations
        super().__init__(
            f"Inference encountered NaN value after {iteration}/{total_iterations} iterations {self.progress}."
        )

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def total_iterations(self) -> int:
        return self._total_iterations

    @property
    def progress(self) -> float:
        return self._iteration / self._total_iterations


class DPVIModel(InferenceModel):

    def __init__(
            self,
            model: ModelFunction,
            output_sample_sites: Iterable[str],
            guide: Optional[GuideFunction] = None
        ) -> None:
        """
        Initialises a probabilistic model for performing differentially-private
        variational inference.

        Args:
            model (ModelFunction): A numpyro model function that programmatically describes the probabilistic model
                which generates the data.
            output_sample_sites (Iterable[str]): Collection of identifiers/names of the sample sites in `model` that
                produce the data. Used to correctly order the columns of the generated synthetic data.
            guide (GuideFunction): Optional numpyro function that programmatically describes the variational approximation
                to the true posterior distribution.
        """
        # TODO: make output_sample_sites optional with the following behaviour:
        # If set to None,
        # the samples sites in `model` that have the `obs` keyword are assumed to produce the output columns
        # in the order of their appearance in the model.

        super().__init__()
        self._model = model
        self._output_sample_sites = output_sample_sites

        if guide is None:
            guide = self.create_default_guide(model)

        self._guide = guide

    @staticmethod
    def create_default_guide(model: ModelFunction) -> GuideFunction:
        return numpyro.infer.autoguide.AutoDiagonalNormal(model)

    def fit(self,
            data: pd.DataFrame,
            rng: d3p.random.PRNGState,
            epsilon: float,
            delta: float,
            clipping_threshold: float,
            num_iter: int,
            q: float,
            silent: bool = False) -> InferenceResult:

        # TODO: this currently assumes that data is fully numeric (i.e., categoricals are numbers, not labels)

        num_data = data.size
        batch_size = int(num_data * q)
        num_epochs = int(num_iter * q)
        dp_scale, _, _ = d3p.dputil.approximate_sigma(epsilon, delta, q, num_iter, maxeval=20)

        optimizer = numpyro.optim.Adam(1e-3)

        svi = d3p.svi.DPSVI(
            self._model, self._guide, optimizer, numpyro.infer.Trace_ELBO(),
            num_obs_total=num_data, clipping_threshold=clipping_threshold,
            dp_scale=dp_scale, rng_suite=d3p.random
        )

        svi_rng, init_batch_rng, epochs_rng = d3p.random.split(rng, 3)

        data = np.asarray(data)
        init_batching, get_batch = d3p.minibatch.subsample_batchify_data((data,), batch_size, rng_suite=d3p.random)
        _, batchify_state = init_batching(init_batch_rng)

        batch = get_batch(0, batchify_state)
        svi_state = svi.init(svi_rng, *batch)

        @jax.jit
        def train_epoch(num_epoch_iter, svi_state, batchify_state):
            def train_iteration(i, state_and_loss):
                svi_state, loss = state_and_loss
                batch = get_batch(i, batchify_state)
                svi_state, iter_loss = svi.update(svi_state, *batch)
                return (svi_state, loss + iter_loss / num_epoch_iter)

            return jax.lax.fori_loop(0, num_epoch_iter, train_iteration, (svi_state, 0.))

        for i in range(num_epochs):
            batchify_rng = d3p.random.fold_in(epochs_rng, i)
            num_batches, batchify_state = init_batching(batchify_rng)

            svi_state, loss = train_epoch(num_batches, svi_state, batchify_state)
            if np.isnan(loss):
                raise InferenceException((i + 1) * num_batches, num_epochs * num_batches)
            loss /= num_data
            if not silent: print("epoch {}: loss {}".format(i, loss))

        params = svi.get_params(svi_state)
        return DPVIResult(
            self._model, self._guide,
            params,
            self._output_sample_sites,
            PrivacyLevel(epsilon, delta, dp_scale),
            loss
        )

    @staticmethod
    def num_iterations_for_epochs(num_epochs: int, subsample_ratio: float) -> int:
        return int(num_epochs / subsample_ratio)

    @staticmethod
    def num_epochs_for_iterations(num_iterations: int, subsample_ratio: float) -> int:
        return int(np.ceil(num_iterations * subsample_ratio))

    @staticmethod
    def batch_size_for_subsample_ratio(subsample_ratio: float, num_data: int) -> int:
        return d3p.minibatch.q_to_batch_size(subsample_ratio, num_data)

    @staticmethod
    def subsample_ratio_for_batch_size(batch_size: int, num_data: int) -> float:
        return d3p.minibatch.batch_size_to_q(batch_size, num_data)
