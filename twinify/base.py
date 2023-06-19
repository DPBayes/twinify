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

import abc
import pandas as pd
from typing import Union, Optional, Iterable, BinaryIO
import d3p.random

class InferenceModel(metaclass=abc.ABCMeta):
    """ A statistical model to generate privacy-preserving synthetic twins data sets from sensitive data. """

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, rng: d3p.random.PRNGState, epsilon: float, delta: float, show_progress: bool, **kwargs) -> 'InferenceResult':
        """ Compute the parameter posterior (approximation) for a given data set, hyperparameters and privacy bounds.

        Args:
            data (pd.DataFrame): A `pandas.DataFrame` containing (sensitive) data.
            rng (d3p.random.PRNGState): A seeded state for the d3p.random secure random number generator.
            epsilon (float): Privacy bound ε.
            delta (float): Privacy bound δ.
            show_progress (bool): Show progress bars.
            kwargs: Optional (model specific) hyperparameters.
        """
        pass


class InferenceResult(metaclass=abc.ABCMeta):
    """ A posterior parameter (approximation) resulting from privacy-preserving inference on a data set
    for a particular `InferenceModel`. """

    @abc.abstractmethod
    def generate(self,
            rng: d3p.random.PRNGState,
            num_parameter_samples: int,
            num_data_per_parameter_sample: int = 1,
            single_dataframe: bool = True) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        """ Samples a number of samples from the parameter posterior (approximation) and generates the given number of
        data points per parameter samples.

        By default returns a single data frame samples from the posterior predictive distribution, i.e.,
        for each data records first a parameter value is drawn from the parameter posterior distribution, then
        the data record is sampled from the model conditioned on that parameter value. `num_parameter_samples`
        in this case determines the number of data records included in the returned data frame.

        This behavior can be customized to sample more than one data record per parameter sample by setting argument
        `num_data_per_parameter_sample` to a value larger than 1, in which case the total number of records
        returned is `num_parameter_samples * num_data_per_parameter_sample`.

        Setting `single_dataframe = False` causes the method to return an iterable collection of data frames,
        each of which contains all data records sampled for a single parameter samples, i.e., in this case
        this method returns `num_parameter_samples` data frames each of containing `num_data_per_parameter_sample`
        records.

        Each of the data frames "looks" like the original data this `InferenceResult` was obtained from,
        i.e., it has identical column names and categorical labels (if any).

        Args:
            - rng: A seeded state for the d3p.random secure random number generator.
            - num_parameter_samples: How often to sample from the parameter posterior approximation.
            - num_data_per_parameter_sample: How many data points to generate for each parameter sample.
            - single_dataframe: Whether to combine data samples into a single data frame or return separate data frames.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement generate_extended yet.")

    @classmethod
    def load(cls, file_path_or_io: Union[str, BinaryIO], **kwargs) -> 'InferenceResult':
        """ Loads an inference result from a file.

        The file can be specified either as a path or an opened file, i.e., both of the following
        are possible:
        ```InferenceResult.load('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'rb') as f:
            InferenceResult.load(f)
        ```.

        Args:
            - file_path_or_io: The file path (as a string) or a `BinaryIO` instance representing the
                file from which to load.
            - kwargs: Optional (model specific) arguments for loading.

        If a `BinaryIO` instance is passed, the cursor position is advanced to after the data representing the
        inference result.

        Exceptions:
            raises an `InvalidFileFormatException` if the data in the file is not a valid representation of the
            inference result type represented by this class.

        Note for subclass implementation:
            Subclasses need only implement an override for `_load_from_io`.
        """
        if isinstance(file_path_or_io, str):
            with open(file_path_or_io, 'rb') as f:
                return cls._load_from_io(f, **kwargs)
        else:
            if not file_path_or_io.readable():
                raise ValueError("file_path_or_io is not a readable BinaryIO instance.")
            return cls._load_from_io(file_path_or_io, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _load_from_io(cls, read_io: BinaryIO, **kwargs) -> 'InferenceResult':
        """ Internal implementation for `load` using a `BinaryIO` instance.

        Args:
            - read_io: A readable `BinaryIO` instance for reading binary data representing the inference result.
            - kwargs: Optional (model specific) arguments for loading.

        Note for subclass implementation:
            This method MUST raise an `InvalidFileFormatException` if the data stored in the file does not represent
            an inference result type implemented by this class (i.e., if the data cannot be loaded as an instance of
            this class).
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load_from_io yet.")

    @classmethod
    def is_file_stored_result(cls, file_path_or_io: Union[str, BinaryIO], **kwargs) -> bool:
        """ Checks whether a file stores data representing the specific inference result type represented by this class.

        The file can be specified either as a path or an opened file, i.e., both of the following
        are possible:
        ```InferenceResult.is_file_stored_result('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'wb') as f:
            InferenceResult.is_file_stored_result(f)
        ```.

        If a `BinaryIO` instance is passed, the cursor position will remain the same after this method returns.

        Args:
            - file_path_or_io: The file path (as a string) or a `BinaryIO` instance representing the
                file to check.
            - kwargs: Optional (model specific) arguments.

        Note for subclass implementation:
            Subclasses need only implement an override for `_is_file_stored_result_from_io`.
        """
        if isinstance(file_path_or_io, str):
            with open(file_path_or_io, 'rb') as f:
                return cls._is_file_stored_result_from_io(f)
        else:
            if not file_path_or_io.readable() or not file_path_or_io.seekable():
                raise ValueError("file_path_or_io is not a readable and seekable BinaryIO instance.")
            return cls._is_file_stored_result_from_io(file_path_or_io)

    @classmethod
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO, **kwargs) -> bool:
        """ Internal implementation for `is_file_stored_result` using a `BinaryIO` instance.

        Args:
            - read_io: A readable and seekable `BinaryIO` instance for reading binary data representing the
                inference result.
            - kwargs: Optional (model specific) arguments.

        Note for subclass implementation:
            Default implementation tries to read the file and returns False if `_load_from_io` raises an
            `InvalidFileFormatException`. Subclasses should implement a more direct way of determining the type of
            inference result stored in the file (e.g. a type identifier in the beginning of the data). Implementation
            should additionally ensure that the implementation does not advance the cursor position of `read_io`.
        """
        try:
            cls._load_from_io(read_io, **kwargs)
            return True
        except InvalidFileFormatException:
            return False

    def store(self, file_path_or_io: Union[str, BinaryIO]) -> None:
        """ Writes the inference result to a file.

        The file can be specified either as a path or an opened file, i.e., both of the following
        are possible:
        ```inference_result.store('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'wb') as f:
            inference_result.store(f)
        ```.

        Args:
            - file_path_or_io: The file path (as a string) or a `BinaryIO` instance representing the
                file to write data into.

        If a `BinaryIO` instance is passed, the cursor position is advanced to after the data representing the
        inference result.

        Note for subclass implementation:
            Subclasses only need to implement an override for `_store_to_io`.
        """
        if isinstance(file_path_or_io, str):
            with open(file_path_or_io, 'wb') as f:
                self._store_to_io(f)
        else:
            if not file_path_or_io.writable():
                raise ValueError("file_path_or_io is not a writable BinaryIO instance.")
            self._store_to_io(file_path_or_io)

    @abc.abstractmethod
    def _store_to_io(self, write_io: BinaryIO) -> None:
        """ Internal implementation for `store` using a `BinaryIO` instance.

        Args:
            - write_io: A writeable `BinaryIO` instance for writing binary data representing the inference result.

        Note for subclass implementation:
            The data written to the file should
            1) not rely overly much on pickling
            2) allow to identify the type of inference result written to the file
                (to identify the class with which to load it; see also `_is_file_stored_result_from_io`)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement store_to_io yet.")


class InvalidFileFormatException(Exception):

    def __init__(self, cls: type, msg: Optional[str] = None) -> None:
        msg = "Details:\n" + msg if msg is not None else ""
        super().__init__(f"The given file cannot be interpreted as {cls}.{msg}")
