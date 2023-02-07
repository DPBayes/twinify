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

import os
import pickle

from typing import BinaryIO, Optional, Dict, Union, Iterable, Tuple, Any

import pandas as pd

import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp
import numpyro

import d3p.random
from twinify.base import InferenceResult, InvalidFileFormatException
from twinify.dpvi import ModelFunction, GuideFunction, PrivacyLevel
from twinify.dpvi.loadable_auto_guide import LoadableAutoGuide
import twinify.serialization
from twinify.dpvi.sampling import sample_synthetic_data
from twinify.dataframe_data import DataDescription


class SamplingException(Exception):
    pass


class DPVIResult(InferenceResult):

    def __init__(self,
            model: ModelFunction,
            guide: GuideFunction,
            parameters: Dict[str, ArrayLike],
            privacy_parameters: PrivacyLevel,
            final_elbo: float,
            data_description: DataDescription,
        ) -> None:

        self._model = model
        self._guide = guide
        self._params = parameters
        self._privacy_level = privacy_parameters
        self._final_elbo = final_elbo
        self._data_description = data_description

    _twinify_model_output_site = '_twinify_output'

    @staticmethod
    def _mark_model_outputs(model: ModelFunction) -> ModelFunction:
        def _model_wrapper(data: Optional[ArrayLike]=None, *args, **kwargs) -> Any:
            """ Wraps a model function and captures the return value in a sampling site named `_twinify_output`,
            which `DPVIResult` uses to read the generated synthetic data from.
            """
            samples = model(*args, **kwargs)
            if len(jnp.shape(samples)) != 2:
                raise SamplingException("A numpyro model for twinify must return the sampled data as a single two-dimensional array.")
            numpyro.deterministic(DPVIResult._twinify_model_output_site, samples)
            return samples
        return _model_wrapper

    def generate(self,
            rng: d3p.random.PRNGState,
            num_parameter_samples: int,
            num_data_per_parameter_sample: int = 1,
            single_dataframe: bool = True
        ) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:

        sampling_rng = d3p.random.convert_to_jax_rng_key(rng)
        samples = sample_synthetic_data(
            self._mark_model_outputs(self._model), self._guide,
            self._params, sampling_rng,
            num_parameter_samples,
            num_data_per_parameter_sample
        )

        samples = samples[self._twinify_model_output_site]

        assert samples.shape[:2] == (num_parameter_samples, num_data_per_parameter_sample)

        def _squash_sample_dims(v: np.array) -> np.array:
            old_shape = np.shape(v)
            new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
            reshaped_v = np.reshape(v, new_shape)
            return reshaped_v

        if single_dataframe:
            samples_df = self._data_description.map_to_categorical(_squash_sample_dims(samples))
            return samples_df
        else:
            return [
                self._data_description.map_to_categorical(samples[i]) for i in range(num_parameter_samples)
            ]

    @classmethod
    def _load_from_io(
            cls, read_io: BinaryIO,
            model: ModelFunction,
            guide: Optional[GuideFunction] = None,
            **kwargs
        ) -> 'InferenceResult':


        parameters, privacy_parameters, final_elbo, data_description, observation_sites =\
            DPVIResultIO.load_params_from_io(read_io)

        if guide is None:
            from twinify.dpvi.dpvi_model import DPVIModel
            guide = LoadableAutoGuide.wrap_for_sampling_and_initialize(
                DPVIModel.create_default_guide(model), observation_sites
            )(model)

        return DPVIResult(model, guide, parameters, privacy_parameters, final_elbo, data_description)

    @classmethod
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO) -> bool:
        return DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=True)

    def _store_to_io(self, write_io: BinaryIO) -> None:
        observation_sites = None
        if isinstance(self._guide, LoadableAutoGuide):
            observation_sites = self._guide.observation_sites

        return DPVIResultIO.store_params_to_io(
            write_io, self._params, self._privacy_level, self._final_elbo, self._data_description, observation_sites
        )

    @property
    def guide(self) -> GuideFunction:
        return self._guide

    @property
    def model(self) -> ModelFunction:
        return self._model

    @property
    def parameters(self) -> Dict[str, ArrayLike]:
        return jax.tree_map(lambda x: np.copy(x), self._params)

    @property
    def privacy_level(self) -> float:
        """ The privacy parameters: epsilon, delta and standard deviation of noise applied during inference. """
        return self._privacy_level

    @property
    def final_elbo(self) -> float:
        """ The final ELBO achieved by the inference (on the training data). """
        return self._final_elbo

    @property
    def data_description(self) -> DataDescription:
        return self._data_description


class DPVIResultIO:

    IDENTIFIER = "DPVI".encode("utf8")
    CURRENT_IO_VERSION = 1
    CURRENT_IO_VERSION_BYTES = CURRENT_IO_VERSION.to_bytes(1, twinify.serialization.ENDIANESS)

    @staticmethod
    # def load_from_io(read_io: BinaryIO, treedef: jax.tree_util.PyTreeDef) -> DPVIResult:
    def load_params_from_io(read_io: BinaryIO) -> Tuple[Dict[str, ArrayLike], Iterable[str]]:
        assert read_io.readable()

        if not DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=False):
            raise InvalidFileFormatException(DPVIResult, "Stored data does not have correct type identifier.")

        current_version = int.from_bytes(read_io.read(1), twinify.serialization.ENDIANESS)
        if current_version != DPVIResultIO.CURRENT_IO_VERSION:
            raise InvalidFileFormatException(DPVIResult, "Stored data uses an unknown storage format version.")

        # parameters = twinify.serialization.read_params(read_io, treedef)
        stored_data = pickle.load(read_io)
        return (
            stored_data['params'],
            stored_data['privacy_level'],
            stored_data['final_elbo'],
            stored_data['data_description'],
            stored_data['observation_sites']
        )

    @staticmethod
    def is_file_stored_result_from_io(read_io: BinaryIO, reset_cursor: bool) -> bool:
        assert read_io.readable()
        assert read_io.seekable()

        identifier = read_io.read(len(DPVIResultIO.IDENTIFIER))
        if reset_cursor:
            read_io.seek(-len(identifier), os.SEEK_CUR)

        if identifier == DPVIResultIO.IDENTIFIER:
            return True

        return False

    @staticmethod
    def store_params_to_io(
            write_io: BinaryIO,
            params: Dict[str, ArrayLike],
            privacy_level: PrivacyLevel,
            final_elbo: float,
            data_description: DataDescription,
            observation_sites: Optional[Iterable[str]]
        ) -> None:

        assert write_io.writable()

        data = {
            'params': params,
            'privacy_level': privacy_level,
            'final_elbo': final_elbo,
            'data_description': data_description,
            'observation_sites': observation_sites,
        }

        write_io.write(DPVIResultIO.IDENTIFIER)
        write_io.write(DPVIResultIO.CURRENT_IO_VERSION_BYTES)
        pickle.dump(data, write_io)
        # twinify.serialization.write_params(params, write_io)
