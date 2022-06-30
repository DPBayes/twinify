import abc
from io import BufferedReader, BufferedWriter
import pandas as pd
from typing import Union, Optional, Iterable
from d3p.random import PRNGState

class InferenceModel(meta=abc.ABCMeta):
    """ A statistical model to generate privacy-preserving synthetic twins data sets from sensitive data. """

    def fit(self, data: pd.DataFrame, rng: PRNGState, epsilon: float, delta: float, **kwargs) -> 'InferenceResult':
        """ Compute the parameter posterior (approximation) for a given data set, hyperparameters and privacy bounds.

        Args:
            data: A `pandas.DataFrame` containing (sensitive) data.
            rng: A seeded state for the d3p.random secure random number generator.
            epsilon: Privacy bound ε.
            delta: Privacy bound δ.
            kwargs: Optional (model specific) hyperparameters.
        """
        pass


class InferenceResult(metaclass=abc.ABCMeta):
    """ A posterior parameter (approximation) resulting from privacy-preserving inference on a data set
    for a particular `InferenceModel`. """

    def generate(self,
            dataset_size: int,
            num_datasets: Optional[int] = 1,
            ) -> Iterable[pd.DataFrame]:
        """ Samples a given number of data sets of the given size.

        Returns an iterable collection of data frames, each representing one generated data set. Each of the data frames
        "looks" like the original data this `InferenceResult` was obtained from, i.e., it has identical column names
        and categorical labels (if any).

        Args:
            - dataset_size: The size of a single generated data set.
            - num_datasets: How many data sets to generate.
        """
        return self.generate_extended(num_datasets, dataset_size, single_dataframe=False)

    def generate_extended(self,
            num_data_per_parameter_sample: int,
            num_parameter_samples: int,
            single_dataframe: Optional[bool] = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
        """ Samples a number of samples from the parameter posterior (approximation) and generates the given number of
        data points per parameter samples.

        By default returns an iterable collection of data frames, each containing the data points samples for a single
        parameter sample. If the `single_dataframe` argument is set to `True`, a single dataframe containing all samples
        is returned. Each of the data frames "looks" like the original data this `InferenceResult` was obtained from,
        i.e., it has identical column names and categorical labels (if any).

        Args:
            - num_data_per_parameter_sample: How many data points to generate for each parameter sample.
            - num_parameter_samples: How often to sample from the parameter posterior (approximation).
            - single_dataframe: Whether to combine data samples into a single data frame or return separate data frames.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement generate_extended yet.")

    @classmethod
    def load(cls, file_path_or_reader: Union[str, BufferedReader]) -> 'InferenceResult':
        """ Loads an inference result from a file.

        The file can be specified either as a path or a reader, i.e., both of the following
        are possible:
        ```InferenceResult.load('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'rb') as f:
            InferenceResult.load(f)
        ```.

        Args:
            - file_path_or_reader: The file path (as a string) or a `BufferedReader` instance representing the
                file from which to load.

        Exceptions:
            raises an `InvalidFileFormatException` if the data in the file is not a valid representation of the
            inference result type represented by this class.

        Note for subclass implementation:
            Default implementation handles opening a `BufferedReader` if a path is given and calls `_load_from_reader`;
            Subclasses need only implement an override for `_load_from_reader`.
        """
        if isinstance(file_path_or_reader, str):
            with open(file_path_or_reader, 'rb') as f:
                return cls._load_from_reader(f)
        else:
            return cls._load_from_reader(file_path_or_reader)

    @classmethod
    def _load_from_reader(cls, file_reader: BufferedReader) -> 'InferenceResult':
        """ Internal implementation for `load` using a `BufferedReader`.

        Args:
            - file_reader: A `BufferedReader` instance for reading binary data representing the inference result.

        Note for subclass implementation:
            This method MUST raise an `InvalidFileFormatException` if the data stored in the file does not represent
            an inference result type implemented by this class (i.e., if the data cannot be loaded as an instance of
            this class).
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load_from_reader yet.")

    @classmethod
    def _is_file_stored_result_from_reader(cls, file_reader: BufferedReader) -> bool:
        """ Internal implementation for `is_file_stored_result_from_reader` using a `BufferedReader`.

        Args:
            - file_reader: A `BufferedReader` instance for reading binary data representing the inference result.

        Note for subclass implementation:
            Default implementation tries to read the file and returns False if `_load_from_reader` raises an
            `InvalidFileFormatException`. Subclasses should implement a more direct way of determining the type of
            inference result stored in the file (e.g. a type identifier in the beginning of the data).
        """
        try:
            cls._load_from_reader(file_reader)
            return True
        except InvalidFileFormatException:
            return False

    @classmethod
    def is_file_stored_result_from_reader(cls, file_path_or_reader: Union[str, BufferedReader]) -> bool:
        """ Checks whether a file stores data representing the specific inference result type represented by this class.

        The file can be specified either as a path or a reader, i.e., both of the following
        are possible:
        ```InferenceResult.is_file_stored_result('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'wb') as f:
            InferenceResult.is_file_stored_result(f)
        ```.

        Args:
            - file_path_or_reader: The file path (as a string) or a `BufferedReader` instance representing the
                file to check.

        Note for subclass implementation:
            Subclasses need only implement an override for `_is_file_stored_result_from_reader`.
        """
        if isinstance(file_path_or_reader, str):
            with open(file_path_or_reader, 'rb') as f:
                return cls._is_file_stored_result_from_reader(f)
        else:
            return cls._is_file_stored_result_from_reader(file_path_or_reader)

    def store(self, file_path_or_writer: Union[str, BufferedWriter]) -> None:
        """ Writes the inference result to a file.

        The file can be specified either as a path or a writer, i.e., both of the following
        are possible:
        ```inference_result.store('my_inference_result.out')```
        or
        ```
        with open('my_inference_result.out', 'wb') as f:
            inference_result.store(f)
        ```.

        Args:
            - file_path_or_writer: The file path (as a string) or a `BufferedWriter` instance representing the
                file to write data into.

        Note for subclass implementation:
            Subclasses only need to implement an override for `_store_to_writer`.
        """
        if isinstance(file_path_or_writer, str):
            with open(file_path_or_writer, 'wb') as f:
                self._store_to_writer(f)
        else:
            self._store_to_writer(file_path_or_writer)

    def _store_to_writer(self, file_writer: BufferedWriter) -> None:
        """ Internal implementation for `store` using a `BufferedWriter`.

        Args:
            - file_writer: A `BufferedWriter` instance for writing binary data representing the inference result.

        Note for subclass implementation:
            The data written to the file should
            1) not rely overly much on pickling
            2) allow to identify the type of inference result written to the file
                (to identify the class with which to load it; see also `_is_file_stored_result_from_reader`)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement store_to_writer yet.")


class InvalidFileFormatException(Exception):

    def __init__(self, cls: type, msg: Optional[str] = None) -> None:
        msg = "Details:\n" + msg if msg is not None else ""
        super().__init__(f"The given file cannot be interpreted as {cls}.{msg}")
