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

import argparse
from typing import List, FrozenSet, Iterable
import pandas as pd
from functools import reduce

import twinify.napsu_mq
import twinify.cli.preprocessing_model as preprocessing_model

class ParsingError(Exception):
    pass


def _parse_model_file(model_file: str) -> Iterable[FrozenSet[str]]:
    with open(model_file, "r") as f:
        return [frozenset(line.split(",")) for line in  f.readlines()]


def load_cli_napsu(
        args: argparse.Namespace, unknown_args: Iterable[str], data_description: twinify.DataDescription
    ) -> twinify.dpvi.DPVIModel:

    column_feature_set = _parse_model_file(args.model_path)

    feature_names = reduce(set.union, column_feature_set, set())
    missing_features = feature_names.difference(data_description.columns)
    if missing_features:
        raise ParsingError(
            "The model specifies features that are not present in the data:\n{}".format(
                ", ".join(missing_features)
            )
        )

    drop_na = args.drop_na

    def preprocess_drop_na_fn(df: pd.DataFrame) -> pd.DataFrame:
        if drop_na:
            df = df.dropna()

        return df

    return preprocessing_model.PreprocessingModel(
        twinify.napsu_mq.NapsuMQModel(column_feature_set),
        preprocess_drop_na_fn
    )
