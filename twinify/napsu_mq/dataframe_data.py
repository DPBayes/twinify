# Copyright 2022 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, List, Union, Mapping
from pandas.api.types import is_integer_dtype, is_categorical_dtype
import torch
import pandas as pd
import itertools
from functools import reduce
from operator import mul
import numpy as np


class DataFrameData:
    """Converter between categorical and integer formatted dataframes."""

    def __init__(self, base_df: pd.DataFrame):
        """Initialise.
        Args:
            base_df (DataFrame): Base categorical dataframe.
        """
        self.base_df = base_df
        self.int_df = self.base_df.copy()
        for col in base_df.columns:
            if is_categorical_dtype(self.base_df[col]):
                self.int_df[col] = self.base_df[col].cat.codes
            elif is_integer_dtype(self.base_df[col]):
                self.int_df[col] = self.base_df[col]
            else:
                raise ValueError(f"DataFrame contains unsupported column type: {self.base_df[col].dtype}")

        self.int_tensor = torch.tensor(self.int_df.values)

        self.n, self.d = base_df.shape

        self.values_by_col = {
            col: list(range(len(self.base_df[col].cat.categories))) if is_categorical_dtype(self.base_df[col])
            else sorted(list(self.base_df[col].unique()))
            for col in self.int_df.columns
        }
        self.values_by_int_feature = {i: list(self.values_by_col[col]) for i, col in enumerate(self.int_df.columns)}

    def get_x_values(self) -> torch.Tensor:
        """Enumerate all possible datapoints for the categories of the dataset.
        Returns:
            torch.tensor: Enumeration of all possible datapoints.
        """
        x_values = torch.zeros((self.get_domain_size(), self.d))
        for i, val in enumerate(itertools.product(*self.values_by_col.values())):
            x_values[i, :] = torch.tensor(val)
        return x_values

    def get_domain_size(self) -> int:
        """Compute the number of possible datapoints in the domain of the dataset.
        Returns:
            int: The number of possible datapoints in the domain.
        """
        return reduce(mul, [len(col_values) for col_values in self.values_by_col.values()])

    @staticmethod
    def get_category_mapping(categorical_df: pd.DataFrame) -> Mapping[str, Mapping[int, str]]:
        """Returns mapping for categorical columns as dictionary
        Args:
            categorical_df (DataFrame): Categorical dataframe
        Returns:
            category_mapping (Mapping): Two-level dictionary for category mapping.
                First level is mapping from column name to dictionary, second level is mapping from category index value to category names.
                Example:
                    {
                        "animal_column": {
                            0: "cat"
                            1: "dog"
                            2: "sheep"
                        },
                        "ml_library_column": {
                            0: "jax",
                            1: "PyTorch"
                            2: "Tensorflow"
                        }
                    }
        """

        category_mapping = dict()

        for column in categorical_df.columns:
            if is_categorical_dtype(categorical_df[column]):
                column_category_mapping: Mapping[int, str] = dict(enumerate(categorical_df[column].cat.categories))
                category_mapping[column] = column_category_mapping

        return category_mapping

    @staticmethod
    def apply_category_mapping(int_df: pd.DataFrame, category_mapping: Mapping) -> pd.DataFrame:
        """
        Apply categorical mapping to integer dataframe produced by get_categorical_mapping method.
        If a column name doesn't exist in the category_mapping, the column is ignored.
        """

        cat_df = int_df.copy()

        for column in int_df:
            if column in category_mapping:
                cat_df[column] = pd.Categorical(int_df[column]).rename_categories(category_mapping[column],
                                                                                  inplace=True)

        return cat_df

    def int_df_to_cat_df(self, int_df: pd.DataFrame) -> pd.DataFrame:
        """Convert interger-valued dataframe to categorical dataframe.
        Args:
            int_df (DataFrame): The interger valued dataframe.
        Returns:
            DataFrame: Categorical valued dataframe.
        """
        cat_df = int_df.copy()
        for col in int_df.columns:
            cat_df[col] = self.base_df[col].cat.categories[int_df[col]]

        return cat_df.astype("category")

    def ndarray_to_cat_df(self, ndarray: np.ndarray) -> pd.DataFrame:
        """Convert integer-valued ndarray to categorical dataframe.
        Args:
            ndarray (ndarray): The interger-valued array to convert.
        Returns:
            DataFrame: The categorical dataframe.
        """
        int_df = pd.DataFrame(ndarray, columns=self.base_df.columns, dtype=int)
        return self.int_df_to_cat_df(int_df)

    def int_query_to_str_query(self, inds: Tuple, value: Union[Tuple, torch.Tensor]) -> Tuple[List, List]:
        """Convert marginal query for integer dataframe to query for categorical dataframe.
        Args:
            inds (tuple: Query indices.
            value (tuple or torch.Tensor): Query value.
        Returns:
            (List, List): String-valued indices and value.
        """
        value = tuple(value.numpy()) if type(value) is torch.Tensor else value
        str_inds = [self.base_df.columns[ind] for ind in inds]
        str_value = [self.base_df[str_inds[i]].cat.categories[val] for i, val in enumerate(value)]
        return str_inds, str_value

    def str_query_to_int_query(self, feature_set: Tuple, value: Tuple) -> Tuple[List, torch.Tensor]:
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

        int_inds = [index(list(self.values_by_col.keys()), feature) for feature in feature_set]
        int_values = [index(list(self.base_df[feature].cat.categories), value[i]) for i, feature in
                      enumerate(feature_set)]
        return int_inds, torch.tensor(int_values)
