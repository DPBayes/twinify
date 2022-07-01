from msilib.schema import Binary
import pandas as pd

from typing import BinaryIO, Optional, Callable, Any, BinaryIO, Dict, Union, Iterable

import d3p.random
import d3p.dputil
import numpyro.infer
import twinify.infer
from twinify.base import InferenceModel, InferenceResult, InvalidFileFormatException
import jax
import os
import twinify.serialization

ModelFunction = Callable
GuideFunction = Callable

class DPVIModel(InferenceModel):

    def __init__(self, model: ModelFunction, guide: GuideFunction) -> None:
        super().__init__()
        self._model = model
        self._guide = guide

    @staticmethod
    def create_with_autoguide(model: ModelFunction) -> 'DPVIModel':
        guide = numpyro.infer.autoguide.AutoDiagonalNormal(model)
        return DPVIModel(model, guide)

    def fit(self,
            data: pd.DataFrame,
            rng: d3p.random.PRNGState,
            epsilon: float,
            delta: float,
            clipping_threshold: float,
            num_iter: int,
            q: float) -> InferenceResult:

        num_data = data.size
        batch_size = int(num_data * q)
        num_epochs = int(num_iter * q)
        dp_scale = d3p.dputil.approximate_sigma(epsilon, delta, q, num_iter)
        params, _ = twinify.infer.train_model(
            rng, d3p.random, self._model, self._guide, data, batch_size, num_data, dp_scale, num_epochs, clipping_threshold
        )
        return DPVIResult(self._model, self._guide, params)


class DPVIResult(InferenceResult):

    def __init__(self, model: ModelFunction, guide: GuideFunction, parameters: Dict[str, Any]) -> None:
        self._model = model
        self._guide = guide
        self._params = parameters

    def generate(self,
            dataset_size: int,
            num_datasets: Optional[int] = 1,
            ) -> Iterable[pd.DataFrame]:
        pass

    def generate_extended(self,
            num_data_per_parameter_sample: int,
            num_parameter_samples: int,
            single_dataframe: Optional[bool] = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        pass

    @classmethod
    def _load_from_io(cls, read_io: BinaryIO) -> 'InferenceResult':
        raise NotImplementedError("need to figure out how to reconstruct or load treedef here")
        treedef = None
        return DPVIResultIO.load_from_io(read_io, treedef)

    @classmethod
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO) -> bool:
        return DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=True)

    def _store_to_io(self, write_io: BinaryIO) -> None:
        return DPVIResultIO.store_to_io(write_io, self._params)


class DPVIResultIO:

    IDENTIFIER = "DPVI".encode("utf8")
    CURRENT_IO_VERSION = 1
    CURRENT_IO_VERSION_BYTES = CURRENT_IO_VERSION.to_bytes(1, twinify.serialization.ENDIANESS)

    @staticmethod
    def load_from_io(read_io: BinaryIO, treedef: jax.tree_util.PyTreeDef) -> DPVIResult:
        assert read_io.readable()

        if not DPVIResultIO.is_file_stored_result_from_io(read_io, reset_cursor=False):
            raise InvalidFileFormatException(DPVIResult, "Stored data does not have correct type identifier.")

        current_version = int.from_bytes(read_io.read(1), twinify.serialization.ENDIANESS)
        if current_version != DPVIResultIO.CURRENT_IO_VERSION:
            raise InvalidFileFormatException(DPVIResult, "Stored data uses an unknown storage format version.")

        parameters = twinify.serialization.read_params(read_io, treedef)

        raise NotImplementedError("need to figure out how to get model and guide here")

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
    def store_to_io(write_io: BinaryIO, params: Any) -> None:
        assert write_io.writable()

        write_io.write(DPVIResultIO.IDENTIFIER)
        write_io.write(DPVIResultIO.CURRENT_IO_VERSION_BYTES)
        twinify.serialization.write_params(params, write_io)
