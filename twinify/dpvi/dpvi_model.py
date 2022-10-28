import pandas as pd

from typing import BinaryIO, Optional, Callable, Any, BinaryIO, Dict, Union, Iterable, Tuple

import os
import pickle
import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp

import d3p.minibatch
import d3p.random
import d3p.dputil
import numpyro.infer
import twinify.infer
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
import twinify.serialization
import twinify.sampling

from twinify.dpvi.dpvi_result import DPVIResult


ModelFunction = Callable
GuideFunction = Callable


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
            q: float) -> InferenceResult:

        # TODO: this currently assumes that data is fully numeric (i.e., categoricals are numbers, not labels)

        num_data = data.size
        batch_size = int(num_data * q)
        num_epochs = int(num_iter * q)
        dp_scale, _, _ = d3p.dputil.approximate_sigma(epsilon, delta, q, num_iter, maxeval=20)
        params, _ = twinify.infer.train_model(
            rng, d3p.random, self._model, self._guide, (data,), batch_size, num_data, dp_scale, num_epochs, clipping_threshold
        )
        return DPVIResult(self._model, self._guide, params, self._output_sample_sites)

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
