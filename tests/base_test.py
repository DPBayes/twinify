import unittest

import os
from typing import Optional, Union, Iterable, BinaryIO
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import d3p.random

from twinify.base import InferenceResult, InvalidFileFormatException

class InferenceResultTests(unittest.TestCase):

    class TestInferenceResult(InferenceResult):

        def __init__(self, magic: Optional[str] = None, **kwargs) -> None:
            self._kwargs = kwargs
            self._magic = magic

        def generate_extended(self,
                rng: d3p.random.PRNGState,
                num_data_per_parameter_sample: int,
                num_parameter_samples: int,
                single_dataframe: Optional[bool] = False) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
            rng = d3p.random.convert_to_jax_rng_key(rng)

            data = jax.random.normal(rng, shape=(num_parameter_samples * num_data_per_parameter_sample, 3))

            if single_dataframe:
                return pd.DataFrame(np.asarray(data))
            else:
                data = np.reshape(np.asarray(data), (num_parameter_samples, num_data_per_parameter_sample, 3))
                return [pd.DataFrame(x) for x in data]

        @classmethod
        def _load_from_io(cls, read_io: BinaryIO, **kwargs) -> InferenceResult:
            id_bytes = read_io.read(2)
            if id_bytes != b"\x00\xff":
                raise InvalidFileFormatException(InferenceResultTests.TestInferenceResult)
            magic = read_io.readline().decode('utf-8')
            return InferenceResultTests.TestInferenceResult(magic, **kwargs)

        def _store_to_io(self, write_io: BinaryIO) -> None:
            write_io.write(b"\x00\xff")
            write_io.write(self._magic.encode('utf-8'))

    def test_generate(self) -> None:
        result = InferenceResultTests.TestInferenceResult()

        num_datasets = 1
        dataset_size = 10
        syn_data_sets = result.generate(d3p.random.PRNGKey(7), dataset_size)
        self.assertIsInstance(syn_data_sets, Iterable)
        self.assertEqual(num_datasets, len(syn_data_sets))
        for syn_data in syn_data_sets:
            self.assertIsInstance(syn_data, pd.DataFrame)
            self.assertEqual(syn_data.shape, (dataset_size, 3))

        num_datasets = 5
        syn_data_sets = result.generate(d3p.random.PRNGKey(7), dataset_size, num_datasets)
        self.assertIsInstance(syn_data_sets, Iterable)
        self.assertEqual(num_datasets, len(syn_data_sets))
        for syn_data in syn_data_sets:
            self.assertIsInstance(syn_data, pd.DataFrame)
            self.assertEqual(syn_data.shape, (dataset_size, 3))

    def test_store_load_file_object(self) -> None:
        magic = "magic"

        result = InferenceResultTests.TestInferenceResult(magic)
        with tempfile.TemporaryFile('w+b') as f:
            f.name
            result.store(f)

            f.seek(0)
            loaded_result = InferenceResultTests.TestInferenceResult.load(f, other_arg=4.)

            self.assertDictEqual({'other_arg': 4.}, loaded_result._kwargs)
            self.assertEqual(magic, loaded_result._magic)


    def test_store_load_file_path(self) -> None:
        magic = "magic2"

        result = InferenceResultTests.TestInferenceResult(magic)
        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name

        try:
            result.store(filename)
            loaded_result = InferenceResultTests.TestInferenceResult.load(filename, other_arg=4.)

            self.assertDictEqual({'other_arg': 4.}, loaded_result._kwargs)
            self.assertEqual(magic, loaded_result._magic)
        except:
            os.remove(filename)

    def test_is_file_stored_result(self) -> None:
        magic = "magic3"

        result = InferenceResultTests.TestInferenceResult(magic)
        with tempfile.TemporaryFile('w+b') as f:
            result.store(f)
            f.seek(0)

            self.assertTrue(InferenceResultTests.TestInferenceResult.is_file_stored_result(f))

        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name

        try:
            result.store(filename)

            self.assertTrue(InferenceResultTests.TestInferenceResult.is_file_stored_result(filename))
        except:
            os.remove(filename)

    def test_is_file_stored_result_invalid_file(self) -> None:
        with tempfile.TemporaryFile('w+b') as f:
            f.write("some other content".encode("utf-8"))
            self.assertFalse(InferenceResultTests.TestInferenceResult.is_file_stored_result(f))

        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name
            f.write("some other content".encode("utf-8"))

        try:
            self.assertFalse(InferenceResultTests.TestInferenceResult.is_file_stored_result(filename))
        except:
            os.remove(filename)

