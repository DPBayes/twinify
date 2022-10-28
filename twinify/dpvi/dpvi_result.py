import os
import pickle

from typing import BinaryIO, Optional, Callable, BinaryIO, Dict, Union, Iterable, Tuple
from collections import namedtuple

import pandas as pd

import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp

import d3p.random
from twinify.base import InferenceResult, InvalidFileFormatException
import twinify.sampling
import twinify.serialization

ModelFunction = Callable
GuideFunction = Callable

PrivacyParameters = namedtuple("PrivacyLevel", ["epsilon", "delta", "dp_noise"])


class DPVIResult(InferenceResult):

    def __init__(self,
            model: ModelFunction,
            guide: GuideFunction,
            parameters: Dict[str, ArrayLike],
            output_sample_sites: Iterable[str],
            privacy_parameters: PrivacyParameters,
            final_elbo: float
        ) -> None:

        self._model = model
        self._guide = guide
        self._params = parameters
        self._output_sample_sites = tuple(output_sample_sites)
        self._privacy_params = privacy_parameters
        self._final_elbo = final_elbo

    def generate_extended(self,
            rng: d3p.random.PRNGState,
            num_data_per_parameter_sample: int,
            num_parameter_samples: int,
            single_dataframe: Optional[bool] = False
        ) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:

        jax_rng = d3p.random.convert_to_jax_rng_key(rng)
        samples = twinify.sampling.sample_synthetic_data(
            self._model, self._guide, self._params, jax_rng, num_parameter_samples, num_data_per_parameter_sample
        )

        def _squash_sample_dims(v: jnp.array) -> jnp.array:
            old_shape = jnp.shape(v)
            assert len(old_shape) >= 2
            new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
            reshaped_v = jnp.reshape(v, new_shape)
            return reshaped_v

        def _sites_to_output_df(samples: Dict[str, jnp.array]) -> pd.DataFrame:
            try:
                output_samples_np = np.hstack([samples[site] for site in self._output_sample_sites])
                output_samples_df = pd.DataFrame(output_samples_np)
                return output_samples_df
            except KeyError:
                # TODO: meaningful error message
                raise

        if single_dataframe:
            samples_df = _sites_to_output_df({k: _squash_sample_dims(v) for k, v in samples.items()})
            return samples_df
        else:
            for i in range(num_parameter_samples):
                local_samples_df = _sites_to_output_df({k: pd.DataFrame(np.asarray(v[i])) for k, v in samples.items()})
                yield local_samples_df
        # TODO: currently performs no post-processing / remapping of integer values to categoricals

    @classmethod
    def _load_from_io(
            cls, read_io: BinaryIO,
            model: ModelFunction,
            guide: Optional[GuideFunction] = None,
            **kwargs
        ) -> 'InferenceResult':

        if guide is None:
            from twinify.dpvi.dpvi_model import DPVIModel
            guide = DPVIModel.create_default_guide(model)

        parameters, output_sample_sites, privacy_parameters, final_elbo = DPVIResultIO.load_params_from_io(read_io)

        return DPVIResult(model, guide, parameters, output_sample_sites, privacy_parameters, final_elbo)

    @classmethod
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO) -> bool:
        return DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=True)

    def _store_to_io(self, write_io: BinaryIO) -> None:
        return DPVIResultIO.store_params_to_io(
            write_io, self._params, self._output_sample_sites, self._privacy_params, self._final_elbo
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
    def privacy_parameters(self) -> float:
        """ The privacy parameters: epsilon, delta and standard deviation of noise applied during inference. """
        return self._privacy_params

    @property
    def final_elbo(self) -> float:
        """ The final ELBO achieved by the inference (on the training data). """
        return self._final_elbo


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
        # raise NotImplementedError("need to figure out how to get model and guide here")
        stored_data = pickle.load(read_io)
        return (
            stored_data['params'],
            stored_data['output_sample_sites'],
            stored_data['privacy_parameters'],
            stored_data['final_elbo'],
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
            output_sample_sites: Iterable[str],
            privacy_parameters: PrivacyParameters,
            final_elbo: float
        ) -> None:

        assert write_io.writable()

        data = {
            'params': params,
            'output_sample_sites': list(output_sample_sites),
            'privacy_parameters': privacy_parameters,
            'final_elbo': final_elbo,
        }

        write_io.write(DPVIResultIO.IDENTIFIER)
        write_io.write(DPVIResultIO.CURRENT_IO_VERSION_BYTES)
        pickle.dump(data, write_io)
        # twinify.serialization.write_params(params, write_io)
