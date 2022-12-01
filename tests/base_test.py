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

    class InferenceResultTestStub(InferenceResult):

        def __init__(self, magic: Optional[str] = None, **kwargs) -> None:
            self._kwargs = kwargs
            self._magic = magic

        def generate(self,
                rng: d3p.random.PRNGState,
                num_data_per_parameter_sample: int,
                num_parameter_samples: int = 1,
                single_dataframe: Optional[bool] = True) -> Union[Iterable[pd.DataFrame], pd.DataFrame]:
            raise NotImplementedError("not implemented for testing")

        @classmethod
        def _load_from_io(cls, read_io: BinaryIO, **kwargs) -> InferenceResult:
            id_bytes = read_io.read(2)
            if id_bytes != b"\x00\xff":
                raise InvalidFileFormatException(InferenceResultTests.InferenceResultTestStub)
            magic = read_io.readline().decode('utf-8')
            return InferenceResultTests.InferenceResultTestStub(magic, **kwargs)

        def _store_to_io(self, write_io: BinaryIO) -> None:
            write_io.write(b"\x00\xff")
            write_io.write(self._magic.encode('utf-8'))

    def test_store_load_file_object(self) -> None:
        magic = "magic"

        result = InferenceResultTests.InferenceResultTestStub(magic)
        with tempfile.TemporaryFile('w+b') as f:
            f.name
            result.store(f)

            f.seek(0)
            loaded_result = InferenceResultTests.InferenceResultTestStub.load(f, other_arg=4.)

            self.assertDictEqual({'other_arg': 4.}, loaded_result._kwargs)
            self.assertEqual(magic, loaded_result._magic)


    def test_store_load_file_path(self) -> None:
        magic = "magic2"

        result = InferenceResultTests.InferenceResultTestStub(magic)
        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name

        try:
            result.store(filename)
            loaded_result = InferenceResultTests.InferenceResultTestStub.load(filename, other_arg=4.)

            self.assertDictEqual({'other_arg': 4.}, loaded_result._kwargs)
            self.assertEqual(magic, loaded_result._magic)
        except:
            os.remove(filename)

    def test_is_file_stored_result(self) -> None:
        magic = "magic3"

        result = InferenceResultTests.InferenceResultTestStub(magic)
        with tempfile.TemporaryFile('w+b') as f:
            result.store(f)
            f.seek(0)

            self.assertTrue(InferenceResultTests.InferenceResultTestStub.is_file_stored_result(f))

        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name

        try:
            result.store(filename)

            self.assertTrue(InferenceResultTests.InferenceResultTestStub.is_file_stored_result(filename))
        except:
            os.remove(filename)

    def test_is_file_stored_result_invalid_file(self) -> None:
        with tempfile.TemporaryFile('w+b') as f:
            f.write("some other content".encode("utf-8"))
            self.assertFalse(InferenceResultTests.InferenceResultTestStub.is_file_stored_result(f))

        with tempfile.NamedTemporaryFile('w+b', delete=False) as f:
            filename = f.name
            f.write("some other content".encode("utf-8"))

        try:
            self.assertFalse(InferenceResultTests.InferenceResultTestStub.is_file_stored_result(filename))
        except:
            os.remove(filename)

