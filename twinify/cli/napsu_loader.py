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

    return twinify.napsu_mq.NapsuMQModel(column_feature_set)