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


import unittest
import itertools
import pandas as pd
import numpy as np
from twinify.dataframe_data import DataFrameData, DataDescription

class DataDescriptionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dtypes = {
            'int': np.dtype(np.int32),
            'low_prec_float': np.dtype(np.float32),
            'high_prec_float': np.dtype(np.float64),
            'strings!': pd.CategoricalDtype(['this is a string', 'this is another string']),
            'cats!': pd.CategoricalDtype(['first_cat', 'second_cat', 'third_cat']),
        }

        ints = np.arange(4, dtype=np.int32)
        low_prec_floats = np.arange(4, dtype=np.float32)
        high_prec_floats = np.linspace(-2, 3.3, num=4, dtype=np.float64)
        strings = np.array([0, 0, 0, 1], dtype=np.int8)
        cats = np.array([0, 2, 2, 0], dtype=np.int8)
        self.cat_and_string_df = pd.DataFrame({
            'int': ints,
            'low_prec_float': low_prec_floats,
            'high_prec_float': high_prec_floats,
            'strings!': np.array([self.dtypes["strings!"].categories[i] for i in strings]),
            'cats!': pd.Categorical.from_codes(cats, dtype=self.dtypes['cats!']),
        })
        self.cat_df = pd.DataFrame({
            'int': ints,
            'low_prec_float': low_prec_floats,
            'high_prec_float': high_prec_floats,
            'strings!': pd.Categorical.from_codes(strings, dtype=self.dtypes['strings!']),
            'cats!': pd.Categorical.from_codes(cats, dtype=self.dtypes['cats!']),
        })
        self.num_df = pd.DataFrame({
            'int': ints,
            'low_prec_float': low_prec_floats,
            'high_prec_float': high_prec_floats,
            'strings!': strings,
            'cats!': cats,
        })

    def test_init(self) -> None:
        data_description = DataDescription(self.dtypes)
        self.assertDictEqual(self.dtypes, data_description.dtypes)
        self.assertEqual(self.dtypes.keys(), data_description.columns)
        self.assertEqual(len(self.dtypes.keys()), data_description.num_columns)
        self.assertFalse(data_description.all_columns_discrete)

    def test_equals(self) -> None:
        dd1 = DataDescription(self.dtypes)
        dd2 = DataDescription(self.dtypes)

        assert dd1 is not dd2

        self.assertEqual(dd1, dd2)

        other_dtypes = self.dtypes.copy()
        other_dtypes['int'] = np.dtype(np.int64)
        dd3 = DataDescription(other_dtypes)
        self.assertNotEqual(dd1, dd3)

        self.assertNotEqual(dd1, object())

    def test_all_columns_discrete(self) -> None:
        dtypes = dict()
        for col in ['int', 'strings!', 'cats!']:
            dtypes[col] = self.dtypes[col]

        data_description = DataDescription(dtypes)
        self.assertEqual(len(dtypes.keys()), data_description.num_columns)
        self.assertTrue(data_description.all_columns_discrete)

    def test_map_to_numeric(self) -> None:
        data_description = DataDescription(self.dtypes)

        num_df = data_description.map_to_numeric(self.cat_and_string_df)

        self.assertEqual(tuple(self.num_df.columns), tuple(num_df.columns))
        self.assertTrue((self.num_df == num_df).all().all())
        self.assertTrue((self.num_df.dtypes == num_df.dtypes).all())

    def test_map_to_categorical(self) -> None:
        data_description = DataDescription(self.dtypes)

        cat_df = data_description.map_to_categorical(self.num_df)

        self.assertEqual(tuple(self.cat_df.columns), tuple(cat_df.columns))
        self.assertTrue((self.cat_df == cat_df).all().all())
        self.assertTrue((self.cat_df.dtypes == cat_df.dtypes).all())

    def test_map_to_categorical_array(self) -> None:
        data_description = DataDescription(self.dtypes)
        num_array = np.asarray(self.num_df)

        cat_df = data_description.map_to_categorical(num_array)

        self.assertEqual(tuple(self.cat_df.columns), tuple(cat_df.columns))
        self.assertTrue((self.cat_df == cat_df).all().all())
        self.assertTrue((self.cat_df.dtypes == cat_df.dtypes).all())

    def test_from_dataframe(self) -> None:
        data_description = DataDescription.from_dataframe(self.cat_and_string_df)

        expected = DataDescription(self.dtypes)

        self.assertEqual(expected, data_description)


class DataFrameDataTest(unittest.TestCase):
    def setUp(self):
        self.col_parts = ["val", "cal", "mal"]
        self.row_parts = [1, 2, 2, 4, 3]
        self.columns = ["col1", "col2", "col3"]
        self.data = [["{}{}".format(name, i) for name in self.col_parts] for i in self.row_parts]
        self.df = pd.DataFrame(self.data, columns=self.columns, dtype="category")
        self.df_data = DataFrameData(self.df)

        self.data2 = [["False", "False"], ["True", "True"], ["False", "True"], ["False", "False"]]
        self.df2 = pd.DataFrame(self.data2, columns=["A", "B"], dtype="category")
        self.df_data2 = DataFrameData(self.df2)

        self.df3 = pd.DataFrame({
            "int_col1": pd.Series([1, 2, 3, 4], dtype='int8'),
            "int_col2": pd.Series([5, 6, 7, 8], dtype='int8'),
            "cat_col1": pd.Series(["cat", "dog", "mouse", "horse"], dtype='category'),
            "cat_col2": pd.Series(["Java", "Python", "Haskell", "Rust"], dtype='category'),
        })
        self.df_data3 = DataFrameData(self.df3)

    def test_int_df_to_cat_df(self):
        converted_df = self.df_data.int_df_to_cat_df(self.df_data.int_df)
        self.assertTrue((converted_df == self.df).all().all())
        self.assertTrue((converted_df.columns == self.df.columns).all())

        converted_df2 = self.df_data3.int_df_to_cat_df(self.df_data3.int_df)
        self.assertTrue((converted_df2 == self.df3).all().all())
        self.assertTrue((converted_df2.columns == self.df3.columns).all())

    def test_int_df_to_cat_df2(self):
        converted_df = self.df_data2.int_df_to_cat_df(self.df_data2.int_df)
        self.assertTrue((converted_df == self.df2).all().all())
        self.assertTrue((converted_df.columns == self.df2.columns).all())

    def test_ndarray_to_cat_df(self):
        converted_df = self.df_data.ndarray_to_cat_df(self.df_data.int_df.values)
        self.assertTrue((converted_df == self.df).all().all())
        self.assertTrue((converted_df.columns == self.df.columns).all())

    def test_values_by_col_correct_key(self):
        for col in self.columns:
            self.assertIn(col, self.df_data.values_by_col.keys())

    def test_values_by_col_correct_values(self):
        for i, col in enumerate(self.columns):
            for row_part in self.row_parts:
                str_value = "{}{}".format(self.col_parts[i], row_part)
                self.assertIn(list(self.df[col].cat.categories).index(str_value), self.df_data.values_by_col[col])

    def test_x_values(self):
        x_values = self.df_data.get_x_values()
        x_value_list = [tuple(x_values[i, :]) for i in range(x_values.shape[0])]
        for val in itertools.product(*self.df_data.values_by_col.values()):
            self.assertIn(val, x_value_list)

    def test_domain_size(self):
        self.assertEqual(self.df_data.get_domain_size(), 4 ** 3)
        self.assertEqual(self.df_data2.get_domain_size(), 2 ** 2)

    def test_int_query_to_str_query(self):

        res_inds = [["col1", "col2"], ["col3", "col2"], ["col2", "col1", "col3"]]
        res_values = [["val2", "cal4"], ["mal4", "cal3"], ["cal1", "val3", "mal1"]]
        q_inds = [[0, 1], [2, 1], [1, 0, 2]]
        q_values = [[1, 3], [3, 2], [0, 2, 0]]
        for q_ind, q_value, res_ind, res_value in zip(q_inds, q_values, res_inds, res_values):
            inds, values = self.df_data.int_query_to_str_query(q_ind, q_value)
            self.assertListEqual(inds, res_ind)
            self.assertListEqual(values, res_value)

    def test_str_query_to_int_query(self):
        q_inds = [["col1", "col2"], ["col3", "col2"], ["col2", "col1", "col3"]]
        q_values = [["val2", "cal4"], ["mal4", "cal3"], ["cal1", "val3", "mal1"]]
        res_inds = [[0, 1], [2, 1], [1, 0, 2]]
        res_values = [[1, 3], [3, 2], [0, 2, 0]]
        for q_ind, q_value, res_ind, res_value in zip(q_inds, q_values, res_inds, res_values):
            inds, values = self.df_data.str_query_to_int_query(q_ind, q_value)
            self.assertListEqual(inds, res_ind)
            self.assertListEqual(list(values), res_value)

    def test_query_str_to_int_and_back(self):
        q_inds = [["col1", "col2"], ["col3", "col2"], ["col2", "col1", "col3"]]
        q_values = [["val2", "cal4"], ["mal4", "cal3"], ["cal1", "val3", "mal1"]]
        for q_ind, q_value, in zip(q_inds, q_values):
            inds, values = self.df_data.str_query_to_int_query(q_ind, q_value)
            inds, values = self.df_data.int_query_to_str_query(inds, values)
            self.assertListEqual(inds, q_ind)
            self.assertListEqual(values, q_value)

    def test_get_categorical_mapping(self):
        categorical_mapping = DataFrameData.get_category_mapping(self.df_data.base_df)

        true_mapping = {
            "col1": {
                0: "val1",
                1: "val2",
                2: "val3",
                3: "val4"
            },
            "col2": {
                0: "cal1",
                1: "cal2",
                2: "cal3",
                3: "cal4"
            },
            "col3": {
                0: "mal1",
                1: "mal2",
                2: "mal3",
                3: "mal4"
            },
        }

        self.assertDictEqual(categorical_mapping, true_mapping)

        categorical_mapping2 = DataFrameData.get_category_mapping(self.df_data2.base_df)

        true_mapping2 = {
            "A": {
                0: "False",
                1: "True"
            },
            "B": {
                0: "False",
                1: "True"
            }
        }

        self.assertDictEqual(categorical_mapping2, true_mapping2)

    def test_apply_categorical_mapping(self):
        categorical_mapping = DataFrameData.get_category_mapping(self.df_data.base_df)
        df_with_categorical_mapping = DataFrameData.apply_category_mapping(self.df_data.int_df, categorical_mapping)
        pd.testing.assert_frame_equal(self.df_data.base_df, df_with_categorical_mapping)

        categorical_mapping2 = DataFrameData.get_category_mapping(self.df_data2.base_df)
        df_with_categorical_mapping2 = DataFrameData.apply_category_mapping(self.df_data2.int_df, categorical_mapping2)
        pd.testing.assert_frame_equal(self.df_data2.base_df, df_with_categorical_mapping2)

    def test_mixed_dataframe_initialization(self):
        true_int_dataframe = pd.DataFrame({
            "int_col1": pd.Series([1, 2, 3, 4], dtype='int8'),
            "int_col2": pd.Series([5, 6, 7, 8], dtype='int8'),
            "cat_col1": pd.Series([0, 1, 3, 2], dtype='int8'),
            "cat_col2": pd.Series([1, 2, 0, 3], dtype='int8'),
        })

        pd.testing.assert_frame_equal(true_int_dataframe, self.df_data3.int_df)


if __name__ == "__main__":
    unittest.main()
