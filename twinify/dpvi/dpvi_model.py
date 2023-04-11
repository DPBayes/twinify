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

import pandas as pd

from typing import Optional
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb

import d3p.minibatch
import d3p.random
import d3p.dputil
import d3p.svi
import numpyro.infer
from twinify.base import InferenceModel
from tqdm import tqdm

from twinify.dpvi import PrivacyLevel, ModelFunction, GuideFunction
from twinify.dpvi.loadable_auto_guide import LoadableAutoGuide
from twinify.dpvi.dpvi_result import DPVIResult
from twinify.dataframe_data import DataDescription

class InferenceException(Exception):

    def __init__(self, iteration: int, total_iterations: int) -> None:
        self._iteration = iteration
        self._total_iterations = total_iterations
        super().__init__(
            f"Inference encountered NaN value after {iteration}/{total_iterations} iterations {self.progress*100:.3f} %."
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


class SilenceableProgressBar:

    def __init__(self, total: int, silent: bool) -> None:
        self._silent = silent
        if not self._silent:
            self._tqdm = tqdm(total=total, desc="epochs")

    @staticmethod
    def static_update(tqdm, silent: bool, loss: float) -> None:
        if not silent:
            tqdm.set_postfix_str(f"ELBO {loss:.3f}")
            tqdm.update(1)

    def update(self, loss: float) -> None:
        self.static_update(self._tqdm, self._silent, loss)

    def update_from_jax(self, loss: float) -> None:
        def _update_callback(loss, transforms):
            self.static_update(self._tqdm, self._silent, loss)

        if not self._silent:
            hcb.id_tap(_update_callback, loss)

    def close(self) -> None:
        if not self._silent:
            self._tqdm.close()


class DPVIModel(InferenceModel):

    DefaultAutoGuideType = numpyro.infer.autoguide.AutoDiagonalNormal

    @staticmethod
    def create_default_guide(model: ModelFunction) -> GuideFunction:
        return DPVIModel.DefaultAutoGuideType(model)

    def __init__(
            self,
            model: ModelFunction,
            guide: Optional[GuideFunction] = None,
            clipping_threshold: float = 1.,
            num_epochs: int = 1000, # TODO: ultimately this should be num_iters for more flexibility, but the whole pipeline is currently geared towards num_epochs
            subsample_ratio: float = 0.01,
        ) -> None:
        """
        Initialises a probabilistic model for performing differentially-private
        variational inference.

        Args:
            model (ModelFunction): A numpyro model function that programmatically describes the probabilistic model
                which generates the data.
            guide (GuideFunction): Optional numpyro function that programmatically describes the variational approximation
                to the true posterior distribution.
            clipping_threshold (float): The maximum allowed norm of per-example gradients.
            num_epochs (int): How many passes to make over the data.
            subsample_ratio (float): How large of a fraction of the data will constitute a single update batch.
        """
        super().__init__()
        self._model = model

        if guide is None:
            guide = LoadableAutoGuide.wrap_for_inference(self.DefaultAutoGuideType)(model)
        else:
            if isinstance(guide, numpyro.infer.autoguide.AutoGuide):
                warnings.warn(
                    "It seems that you are using an AutoGuide instance, which may result in problems after loading a stored inference model."
                    "Consider wrapping it with twinify.dpvi.LoadableAutoGuide."
                )

        self._guide = guide
        self._clipping_threshold = clipping_threshold
        self._num_epochs = num_epochs
        self._subsample_ratio = subsample_ratio

    def fit(self,
            data: pd.DataFrame,
            rng: d3p.random.PRNGState,
            epsilon: float,
            delta: float,
            silent: bool = False,
            verbose: bool = False) -> DPVIResult:

        q = self._subsample_ratio
        num_epochs = self._num_epochs

        num_data = data.shape[0]
        batch_size = np.maximum(self.batch_size_for_subsample_ratio(q, num_data), 1)
        q = np.maximum(self.subsample_ratio_for_batch_size(batch_size, num_data), q)
        num_iter = self.num_iterations_for_epochs(num_epochs, q)
        dp_scale, act_epsilon, _ = d3p.dputil.approximate_sigma(epsilon, delta, q, num_iter, maxeval=20)

        if verbose:
            scale_per_sample = dp_scale / batch_size
            print(f"Applying noise with std deviation {dp_scale:.2f} (~ {scale_per_sample:.2f} per element in batch)"
                f" to achieve privacy epsilon of {act_epsilon:.3f} (for delta {delta:.2e}). "
            )
            # TODO: warn for high noise? but when is it too high? what is a good heuristic?


        optimizer = numpyro.optim.Adam(1e-3)

        svi = d3p.svi.DPSVI(
            self._model, self._guide,
            optimizer, numpyro.infer.Trace_ELBO(),
            num_obs_total=num_data, clipping_threshold=self._clipping_threshold,
            dp_scale=dp_scale, rng_suite=d3p.random
        )

        svi_rng, init_batch_rng, epochs_rng = d3p.random.split(rng, 3)

        data_description = DataDescription.from_dataframe(data)

        data = np.asarray(data_description.map_to_numeric(data))
        init_batching, get_batch = d3p.minibatch.subsample_batchify_data((data,), batch_size, rng_suite=d3p.random)
        _, batchify_state = init_batching(init_batch_rng)

        batch = get_batch(0, batchify_state)
        svi_state = svi.init(svi_rng, *batch)

        bar = SilenceableProgressBar(num_epochs, silent)

        def is_nan_in_state(svi_state) -> bool:
            params = svi.get_params(svi_state)
            return jnp.logical_not(
                jnp.all(jnp.array(
                    jax.tree_util.tree_leaves(
                        jax.tree_util.tree_map(
                            lambda x: jnp.logical_not(jnp.any(jnp.isnan(x))),
                            params
                        )
                    )
                ))
            )

        @jax.jit
        def loop_cond(args):
            """ While loop condition. Stop if:
                - have reached desired number of epochs
                - any inferred parameter is NaN, indicating failure -> abort
            """
            cur_epoch, svi_state, _ = args
            no_nans = jnp.logical_not(is_nan_in_state(svi_state))
            return jnp.logical_and(cur_epoch < num_epochs, no_nans)

        @jax.jit
        def train_epoch(args):
            e, svi_state, _ = args

            batchify_rng = d3p.random.fold_in(epochs_rng, e)
            num_batches, batchify_state = init_batching(batchify_rng)

            def train_iteration(i, state_and_loss):
                svi_state, loss = state_and_loss
                batch = get_batch(i, batchify_state)
                svi_state, iter_loss = svi.update(svi_state, *batch)
                return (svi_state, loss + iter_loss / (num_batches * num_data))

            new_svi_state, new_loss = jax.lax.fori_loop(0, num_batches, train_iteration, (svi_state, 0.))

            bar.update_from_jax(new_loss)

            return e + 1, new_svi_state, new_loss

        final_epoch, svi_state, loss = jax.lax.while_loop(loop_cond, train_epoch, (0, svi_state, 0.))

        if is_nan_in_state(svi_state):
            raise InferenceException(final_epoch + 1, num_epochs)

        bar.close()

        params = svi.get_params(svi_state)
        return DPVIResult(
            self._model, self._guide,
            params,
            PrivacyLevel(epsilon, delta, dp_scale),
            loss,
            data_description
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
