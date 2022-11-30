import numpy as np
import pandas as pd
import pickle
from typing import Optional, Union, Iterable, BinaryIO

import d3p.random
from twinify.base import InferenceModel, InferenceResult
from twinify import DataDescription
from twinify.cli.dpvi_numpyro_model_loading import TGuardedPreprocessFunction, TGuardedPostprocessFunction


class PostprocessingResult(InferenceResult):

    def __init__(self,
        base_result: InferenceResult,
        postprocessing_fn: TGuardedPostprocessFunction,
        data_description: DataDescription
    ) -> None:

        self._base_result = base_result
        self._postprocessing_fn = postprocessing_fn
        self._data_description = data_description
        super().__init__()

    @property
    def base(self) -> InferenceResult:
        return self._base_result

    @property
    def postprocess_fn(self) -> Optional[TGuardedPostprocessFunction]:
        return self._postprocessing_fn

    def postprocess(self, x: np.ndarray) -> pd.DataFrame:
        if self.postprocess_fn is not None:
            return self.postprocess_fn(x)
        return x

    def generate(self,
            rng: d3p.random.PRNGState,
            num_parameter_samples: int,
            num_data_per_parameter_sample: int = 1,
            single_dataframe: bool = True) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        x = self._base_result.generate(rng, num_parameter_samples, num_data_per_parameter_sample, single_dataframe)
        x_post = self.postprocess(x)
        return x_post

    def _store_to_io(self, write_io: BinaryIO) -> None:
        base_type = type(self._base_result)
        pickle.dump(base_type, write_io)

        self._base_result._store_to_io(write_io)

        pickle.dump(self._data_description, write_io)


    @classmethod
    def _load_from_io(cls, read_io: BinaryIO, postprocessing_fn: TGuardedPostprocessFunction, **kwargs) -> InferenceResult:
        assert read_io.readable()

        base_type = pickle.load(read_io)
        base_result: InferenceResult = base_type._load_from_io(read_io, **kwargs)

        data_description: DataDescription = pickle.load(read_io)

        return PostprocessingResult(base_result, postprocessing_fn, data_description)


class PreprocessingModel(InferenceModel):

    def __init__(self,
        base_model: InferenceModel,
        preprocess_fn: Optional[TGuardedPreprocessFunction] = None,
        postprocess_fn: Optional[TGuardedPostprocessFunction] = None
    ) -> None:

        self._base_model = base_model
        self._preprocess_fn = preprocess_fn
        self._postprocess_fn = postprocess_fn

        super().__init__()

    @property
    def base(self) -> None:
        return self._base_model

    @property
    def preprocess_fn(self) -> Optional[TGuardedPostprocessFunction]:
        return self._preprocess_fn

    @property
    def postprocess_fn(self) -> Optional[TGuardedPostprocessFunction]:
        return self._postprocess_fn

    def preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._preprocess_fn(x)

    def fit(self, data: pd.DataFrame, rng: d3p.random.PRNGState, epsilon: float, delta: float, **kwargs) -> InferenceResult:
        if self._preprocess_fn is not None:
            data = self._preprocess_fn(data.copy())

        result = self.base.fit(data, rng, epsilon, delta, **kwargs)

        if self._postprocess_fn is not None:
            data_description = DataDescription.from_dataframe(data)
            return PostprocessingResult(result, self._postprocess_fn, data_description)

        return result
