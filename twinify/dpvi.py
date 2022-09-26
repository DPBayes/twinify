import pandas as pd

from typing import BinaryIO, Optional, Callable, Any, BinaryIO, Dict, Union, Iterable

import os
import pickle
import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp

import d3p.random
import d3p.dputil
import numpyro.infer
import twinify.infer
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
import twinify.serialization
import twinify.sampling


ModelFunction = Callable
GuideFunction = Callable


class DPVIModel(InferenceModel):

    def __init__(self, model: ModelFunction, guide: Optional[GuideFunction] = None) -> None:
        super().__init__()
        self._model = model

        if guide is None:
            guide = self.create_default_guide(model)

        self._guide = guide

    @classmethod
    @property
    def default_guide_class(cls):
        return numpyro.infer.autoguide.AutoDiagonalNormal

    @staticmethod
    def create_default_guide(model: ModelFunction) -> GuideFunction:
        return DPVIModel.default_guide_class(model)

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
        dp_scale = d3p.dputil.approximate_sigma(epsilon, delta, q, num_iter)
        params, _ = twinify.infer.train_model(
            rng, d3p.random, self._model, self._guide, data, batch_size, num_data, dp_scale, num_epochs, clipping_threshold
        )
        return DPVIResult(self._model, self._guide, params)


class DPVIResult(InferenceResult):

    def __init__(self, model: ModelFunction, guide: GuideFunction, parameters: Dict[str, ArrayLike]) -> None:
        self._model = model
        self._guide = guide
        self._params = parameters

    def generate_extended(self,
            rng: d3p.random.PRNGState,
            num_data_per_parameter_sample: int,
            num_parameter_samples: int,
            single_dataframe: Optional[bool] = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:

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

        if single_dataframe:
            samples = {k: _squash_sample_dims(v) for k, v in samples.items()}
            return pd.DataFrame(samples)
        else:
            for i in range(num_parameter_samples):
                local_samples = {k: v[i] for k, v in samples.items()}
                yield pd.DataFrame(local_samples)

        # TODO: currently performs no post-processing / remapping of integer values to categoricals

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO, model: ModelFunction, guide: Optional[GuideFunction] = None, **kwargs) -> 'InferenceResult':
        if guide is None:
            guide = DPVIModel.create_default_guide(model)

        parameters = DPVIResultIO.load_params_from_io(read_io)

        return DPVIResult(model, guide, parameters)

    @classmethod
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO) -> bool:
        return DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=True)

    def _store_to_io(self, write_io: BinaryIO) -> None:
        return DPVIResultIO.store_params_to_io(write_io, self._params)

    @property
    def guide(self) -> GuideFunction:
        return self._guide

    @property
    def model(self) -> ModelFunction:
        return self._model

    @property
    def parameters(self) -> Dict[str, ArrayLike]:
        return jax.tree_map(lambda x: np.copy(x), self._params)

class DPVIResultIO:

    IDENTIFIER = "DPVI".encode("utf8")
    CURRENT_IO_VERSION = 1
    CURRENT_IO_VERSION_BYTES = CURRENT_IO_VERSION.to_bytes(1, twinify.serialization.ENDIANESS)

    @staticmethod
    # def load_from_io(read_io: BinaryIO, treedef: jax.tree_util.PyTreeDef) -> DPVIResult:
    def load_params_from_io(read_io: BinaryIO) -> Dict[str, ArrayLike]:
        assert read_io.readable()

        if not DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=False):
            raise InvalidFileFormatException(DPVIResult, "Stored data does not have correct type identifier.")

        current_version = int.from_bytes(read_io.read(1), twinify.serialization.ENDIANESS)
        if current_version != DPVIResultIO.CURRENT_IO_VERSION:
            raise InvalidFileFormatException(DPVIResult, "Stored data uses an unknown storage format version.")

        # parameters = twinify.serialization.read_params(read_io, treedef)
        # raise NotImplementedError("need to figure out how to get model and guide here")
        return pickle.load(read_io)

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
    def store_params_to_io(write_io: BinaryIO, params: Dict[str, ArrayLike]) -> None:
        assert write_io.writable()

        write_io.write(DPVIResultIO.IDENTIFIER)
        write_io.write(DPVIResultIO.CURRENT_IO_VERSION_BYTES)
        pickle.dump(params, write_io)
        # twinify.serialization.write_params(params, write_io)
