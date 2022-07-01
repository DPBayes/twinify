import abc
import pandas as pd
from typing import Union, Optional, Iterable, BinaryIO
from d3p.random import PRNGState

class InferenceModel(meta=abc.ABCMeta):
    """ A statistical model to generate privacy-preserving synthetic twins data sets from sensitive data. """

    @abc.abstractmethod
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
            rng: PRNGState,
            dataset_size: int,
            num_datasets: Optional[int] = 1,
            ) -> Iterable[pd.DataFrame]:
        """ Samples a given number of data sets of the given size.

        Returns an iterable collection of data frames, each representing one generated data set. Each of the data frames
        "looks" like the original data this `InferenceResult` was obtained from, i.e., it has identical column names
        and categorical labels (if any).

        Args:
            - rng: A seeded state for the d3p.random secure random number generator.
            - dataset_size: The size of a single generated data set.
            - num_datasets: How many data sets to generate.
        """
        return self.generate_extended(rng, num_datasets, dataset_size, single_dataframe=False)

    @abc.abstractmethod
    def generate_extended(self,
            rng: PRNGState,
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
            - rng: A seeded state for the d3p.random secure random number generator.
            - num_data_per_parameter_sample: How many data points to generate for each parameter sample.
            - num_parameter_samples: How often to sample from the parameter posterior (approximation).
            - single_dataframe: Whether to combine data samples into a single data frame or return separate data frames.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement generate_extended yet.")

    @classmethod
    def load(cls, file_path_or_io: Union[str, BinaryIO]) -> 'InferenceResult':
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
                return cls._load_from_io(f)
        else:
            if not file_path_or_io.readable():
                raise ValueError("file_path_or_io is not a readable BinaryIO instance.")
            return cls._load_from_io(file_path_or_io)

    @classmethod
    @abc.abstractmethod
    def _load_from_io(cls, read_io: BinaryIO) -> 'InferenceResult':
        """ Internal implementation for `load` using a `BinaryIO` instance.

        Args:
            - read_io: A readable `BinaryIO` instance for reading binary data representing the inference result.

        Note for subclass implementation:
            This method MUST raise an `InvalidFileFormatException` if the data stored in the file does not represent
            an inference result type implemented by this class (i.e., if the data cannot be loaded as an instance of
            this class).
        """
        raise NotImplementedError(f"{cls.__name__} does not implement load_from_io yet.")

    @classmethod
    def is_file_stored_result(cls, file_path_or_io: Union[str, BinaryIO]) -> bool:
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
    def _is_file_stored_result_from_io(cls, read_io: BinaryIO) -> bool:
        """ Internal implementation for `is_file_stored_result` using a `BinaryIO` instance.

        Args:
            - read_io: A readable and seekable `BinaryIO` instance for reading binary data representing the
                inference result.

        Note for subclass implementation:
            Default implementation tries to read the file and returns False if `_load_from_io` raises an
            `InvalidFileFormatException`. Subclasses should implement a more direct way of determining the type of
            inference result stored in the file (e.g. a type identifier in the beginning of the data). Implementation
            should additionally ensure that the implementation does not advance the cursor position of `read_io`.
        """
        try:
            cls._load_from_io(read_io)
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
