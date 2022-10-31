import os
import pickle

from typing import BinaryIO, Optional, Dict, Union, Iterable, Tuple

import pandas as pd

import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp
import numpyro

import d3p.random
from twinify.base import InferenceResult, InvalidFileFormatException
from twinify.dpvi import ModelFunction, GuideFunction, PrivacyLevel
import twinify.serialization
from twinify.dpvi.sampling import sample_synthetic_data


class DPVIResult(InferenceResult):

    def __init__(self,
            model: ModelFunction,
            guide: GuideFunction,
            parameters: Dict[str, ArrayLike],
            output_sample_sites: Iterable[str],
            privacy_parameters: PrivacyLevel,
            final_elbo: float
        ) -> None:

        self._model = model
        self._guide = guide
        self._params = parameters
        self._output_sample_sites = tuple(output_sample_sites)
        self._privacy_level = privacy_parameters
        self._final_elbo = final_elbo

    def generate_extended(self,
            rng: d3p.random.PRNGState,
            num_data_per_parameter_sample: int,
            num_parameter_samples: int,
            single_dataframe: Optional[bool] = False
        ) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:

        # @jax.jit
        # def sample_from_ppd(rng_key):
        #     """ Samples a single parameter vector and
        #         num_record_samples_per_parameter_sample based on it.
        #     """
        #     parameter_sampling_rng, record_sampling_rng = jax.random.split(rng_key)

        #     # sample single parameter vector
        #     posterior_sampler = numpyro.infer.Predictive(
        #         self._guide, params=self._params, num_samples=1
        #     )
        #     posterior_samples = posterior_sampler(parameter_sampling_rng)
        #     # models always add a superfluous batch dimensions, squeeze it
        #     posterior_samples = {k: v.squeeze(0) for k,v in posterior_samples.items()}

        #     # sample num_data_per_parameter_sample data samples
        #     ppd_sampler = numpyro.infer.Predictive(self._model, posterior_samples, batch_ndims=0)
        #     per_sample_rngs = jax.random.split(
        #         record_sampling_rng, num_data_per_parameter_sample
        #     )
        #     ppd_samples = jax.vmap(ppd_sampler)(per_sample_rngs)
        #     # models always add a superfluous batch dimensions, squeeze it
        #     ppd_samples = {k: v.squeeze(1) for k, v in ppd_samples.items()}

        #     return ppd_samples

        # sampling_rng = d3p.random.convert_to_jax_rng_key(rng)
        # per_parameter_rngs = jax.random.split(sampling_rng, num_parameter_samples)
        # ppd_samples = jax.vmap(sample_from_ppd)(per_parameter_rngs)

        # samples = {site: np.asarray(value) for site, value in ppd_samples.items()}
        sampling_rng = d3p.random.convert_to_jax_rng_key(rng)
        samples = sample_synthetic_data(
            self._model, self._guide, self._params, sampling_rng, num_parameter_samples, num_data_per_parameter_sample
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
            return [
                _sites_to_output_df({k: pd.DataFrame(v[i]) for k, v in samples.items()}) for i in range(num_parameter_samples)
            ]
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
            write_io, self._params, self._output_sample_sites, self._privacy_level, self._final_elbo
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
            stored_data['privacy_level'],
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
            privacy_level: PrivacyLevel,
            final_elbo: float
        ) -> None:

        assert write_io.writable()

        data = {
            'params': params,
            'output_sample_sites': list(output_sample_sites),
            'privacy_level': privacy_level,
            'final_elbo': final_elbo,
        }

        write_io.write(DPVIResultIO.IDENTIFIER)
        write_io.write(DPVIResultIO.CURRENT_IO_VERSION_BYTES)
        pickle.dump(data, write_io)
        # twinify.serialization.write_params(params, write_io)
