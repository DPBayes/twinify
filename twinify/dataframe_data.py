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

from typing import Tuple, List, Union, Mapping, Iterable, Any, Dict, TypeVar
from pandas.api.types import is_integer_dtype, is_categorical_dtype, is_float_dtype, is_string_dtype, is_bool_dtype
import pandas as pd
import itertools
from functools import reduce
from operator import mul
import numpy as np
import logging

logger = logging.getLogger(__name__)

Dtype = TypeVar("Dtype")


class DataDescription:

    def __init__(self, dtypes: Mapping[str, Dtype]) -> None:
        self._dtypes = dict(dtypes)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, strings_to_categories: bool = True) -> "DataDescription":
        dtypes = dict()
        for col in df.columns:
            dtype = df[col].dtype
            if is_string_dtype(dtype):
                if not strings_to_categories:
                    raise ValueError(
                        f"Input may not contain columns of dtype string unless strings_to_categories is True. Column: {col}.")

                try:
                    dtype = pd.CategoricalDtype(df[col].dropna().unique())
                except ValueError as e:
                    raise Exception(f"Cannot interpet type of column {col}", e)
            if not (is_categorical_dtype(dtype) or is_integer_dtype(dtype) or is_float_dtype(dtype) or is_bool_dtype(dtype)):
                raise ValueError(f"Only float, integer or categorical dtypes are currently supported, but column {col} has dtype {dtype}.")

            dtypes[col] = dtype

        return DataDescription(dtypes)

    @property
    def columns(self) -> Tuple[str]:
        return tuple(self._dtypes.keys())

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def dtypes(self) -> Dict[str, Dtype]:
        return self._dtypes.copy()

    @property
    def all_columns_discrete(self) -> bool:
        return np.all([
            not is_float_dtype(dtype) for dtype in self._dtypes.values()
        ])

    @staticmethod
    def _map_column_to_numeric(x: pd.Series) -> pd.Series:
        assert (
                is_categorical_dtype(x.dtype) or
                is_float_dtype(x.dtype) or
                is_integer_dtype(x.dtype) or
                is_bool_dtype(x.dtype)
        )
        if is_categorical_dtype(x.dtype):
            return x.cat.codes
        return x

    def map_to_numeric(self, categorical_df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = categorical_df.copy()
        for col, dtype in self._dtypes.items():
            if col not in categorical_df.columns:
                continue

            numeric_df[col] = self._map_column_to_numeric(numeric_df[col].astype(dtype))
        return numeric_df

    @staticmethod
    def _are_all_values_integers(x: pd.Series) -> bool:
        return np.allclose(x - x.astype(int), 0.)

    def map_to_categorical(self, numeric_df_or_array: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(numeric_df_or_array, pd.DataFrame):
            numeric_df = numeric_df_or_array
        else:
            shape = np.shape(numeric_df_or_array)
            if len(shape) != 2 or shape[1] != len(self.columns):
                raise ValueError(
                    f"Array must be two-dimensional with a second dimension of size {len(self.columns)}, had shape {shape}.")
            numeric_df = pd.DataFrame(numeric_df_or_array, columns=self.columns)

        categorical_df = pd.DataFrame()

        for col, dtype in self._dtypes.items():
            if col not in numeric_df.columns:
                continue

            col_values = numeric_df[col]
            if is_categorical_dtype(dtype):
                if not self._are_all_values_integers(col_values):
                    raise ValueError(f"Cannot map column {col} to categories because the input were no integers.")

                categorical_df[col] = pd.Categorical.from_codes(col_values.astype(int), dtype=dtype)
            else:
                categorical_df[col] = col_values.astype(dtype)
        return categorical_df

    def __eq__(self, other) -> bool:
        return isinstance(other, DataDescription) and self._dtypes == other._dtypes


class DataFrameData:
    """Converter between categorical and integer formatted dataframes."""

    def __init__(self, base_df: pd.DataFrame):
        """Initialise.
        Args:
            base_df (DataFrame): Base categorical dataframe.
        """
        self._data_description = DataDescription.from_dataframe(base_df)
        if not self._data_description.all_columns_discrete:
            raise ValueError(
                "Data has columns which are not discrete, i.e., neither of type bool, integer, or categorical.")

        self.int_df = self._data_description.map_to_numeric(base_df)

        self._values_by_col = {
            col: list(range(len(base_df[col].cat.categories))) if is_categorical_dtype(base_df[col])
            else sorted(list(base_df[col].unique()))
            for col in self.int_df.columns
            if (is_categorical_dtype(self.int_df[col]) or is_integer_dtype(self.int_df[col]))
        }
        # self.values_by_int_feature = { #TODO: rename to values_by_feature_idx if used anywhere?
        #     i: list(self._values_by_col[col])
        #     for i, col in enumerate(self.int_df.columns)
        #     if (is_categorical_dtype(self.int_df[col]) or is_integer_dtype(self.int_df[col]))
        # }

    @property
    def values_by_col(self) -> Dict[str, List[int]]:
        return self._values_by_col

    @property
    def columns(self) -> Iterable[str]:
        return self.int_df.columns

    @property
    def data_description(self) -> DataDescription:
        return self._data_description

    def get_x_values(self) -> np.ndarray:
        """Enumerate all possible datapoints for the categories of the dataset.
        Returns:
            np.ndarray: Enumeration of all possible datapoints.
        """
        x_values = np.zeros((self.get_domain_size(), len(self.columns)))
        for i, val in enumerate(itertools.product(*self.values_by_col.values())):
            x_values[i, :] = np.array(val)
        return x_values

    def get_domain_size(self) -> int:
        """Compute the number of possible datapoints in the domain of the dataset.
        Returns:
            int: The number of possible datapoints in the domain.
        """
        return reduce(mul, [len(col_values) for col_values in self.values_by_col.values()])

    def int_df_to_cat_df(self, int_df: pd.DataFrame) -> pd.DataFrame:
        """Convert integer-valued dataframe to categorical dataframe.
        Args:
            int_df (DataFrame): The integer valued dataframe.
        Returns:
            DataFrame: Categorical valued dataframe.
        """
        return self._data_description.map_to_categorical(int_df)

    def ndarray_to_cat_df(self, ndarray: np.ndarray) -> pd.DataFrame:
        """Convert integer-valued ndarray to categorical dataframe.
        Args:
            ndarray (ndarray): The integer-valued array to convert.
        Returns:
            DataFrame: The categorical dataframe.
        """
        return self._data_description.map_to_categorical(ndarray)

    def int_query_to_str_query(self, inds: Iterable[int], value: Iterable[int]) -> Tuple[List[str], List[str]]:
        """Convert marginal query for integer dataframe to query for categorical dataframe.
        Args:
            inds (tuple: Query indices.
            value (tuple or np.ndarray): Query value.
        Returns:
            (List, List): String-valued indices and value.
        """
        value = tuple(value) if isinstance(value, np.ndarray) else value
        column_names = [self._data_description.columns[ind] for ind in inds]
        str_values = [
            self._data_description.dtypes[column_names[i]].categories[v]
            if is_categorical_dtype(self._data_description.dtypes[column_names[i]])
            else v
            for i, v in enumerate(value)
        ]
        return column_names, str_values

    def str_query_to_int_query(self, feature_set: Iterable[str], value: List[str]) -> Tuple[List, np.ndarray]:
        """Convert marginal query for categorical dataframe to query for integer dataframe.
        Args:
            feature_set (tuple): Query indices.
            value (tuple): Query values.
        Returns:
            (tuple, tuple): Converted query indices and value.
        """

        def index(list, value):
            if value in list:
                return list.index(value)
            else:
                raise ValueError("{} not in {}".format(value, list))

        int_inds = [index(self._data_description.columns, feature) for feature in feature_set]
        int_values = [
            pd.Categorical(value[i], dtype=self._data_description.dtypes[feature]).codes
            if is_categorical_dtype(self._data_description.dtypes[feature])
            else value[i]
            for i, feature in enumerate(feature_set)
        ]
        return int_inds, np.array(int_values)
