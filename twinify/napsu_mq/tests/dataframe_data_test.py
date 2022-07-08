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
from twinify.napsu_mq.dataframe_data import DataFrameData


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

    def test_int_df_to_cat_df(self):
        converted_df = self.df_data.int_df_to_cat_df(self.df_data.int_df)
        self.assertTrue((converted_df == self.df).all().all())
        self.assertTrue((converted_df.columns == self.df.columns).all())

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


if __name__ == "__main__":
    unittest.main()