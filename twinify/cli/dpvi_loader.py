import argparse
from typing import List, Tuple, Callable
import pandas as pd
import numpyro.infer.autoguide

import twinify
import twinify.dpvi
import twinify.dpvi.modelling.automodel as automodel
import twinify.cli.dpvi_numpyro_model_loading as model_loading
import twinify.cli.preprocessing_model as preprocessing_model

def _load_cli_dpvi_automodel(
        model_path: str,
        mixture_components: int,
        clipping_threshold: 1.,
        num_epochs: int,
        subsample_ratio: float,
        drop_na: bool
    ) -> twinify.dpvi.DPVIModel:

    with open(model_path, 'r') as f:
        model_str = "".join(f.readlines())

    features = automodel.parse_model(model_str)

    # build model
    model_fn = automodel.make_model(features, mixture_components)

    # build variational guide for optimization
    guide_fn = numpyro.infer.autoguide.AutoDiagonalNormal(model_fn)

    # postprocessing for automodel
    def preprocess_fn(df: pd.DataFrame) -> pd.DataFrame:

        # pick features from data according to model file
        feature_names = [feature.name for feature in features]
        missing_features = set(feature_names).difference(df.columns)
        if missing_features:
            raise automodel.ParsingError(
                "The model specifies features that are not present in the data:\n{}".format(
                    ", ".join(missing_features)
                )
            )

        df = df.loc[:, feature_names]

        if drop_na:
            df = df.dropna()

        for feature in features:
            df = feature.preprocess_data(df)

        print(f"After preprocessing the data has {df.shape[0]} entries with {df.shape[1]} features each.")

        return df

    postprocess_fn = automodel.postprocess_function_factory(features)

    model = twinify.dpvi.DPVIModel(
        model_fn, guide_fn,
        clipping_threshold,
        num_epochs,
        subsample_ratio
    )

    model = preprocessing_model.PreprocessingModel(
        model, preprocess_fn, postprocess_fn
    )

    return model


def load_cli_dpvi(
        args: argparse.Namespace, unknown_args: List[str], data_description: twinify.DataDescription
    ) -> twinify.dpvi.DPVIModel:

    if args.model_path[-3:] == '.py':
        try:
            model_fn, guide_fn, preprocess_fn, postprocess_fn = model_loading.load_custom_numpyro_model(
                args.model_path, args, unknown_args, data_description
            )

            model = twinify.dpvi.DPVIModel(
                model_fn, guide_fn,
                args.clipping_threshold,
                args.num_epochs,
                args.sampling_ratio
            )

            drop_na = args.drop_na

            def preprocess_drop_na_fn(df: pd.DataFrame) -> pd.DataFrame:
                df = preprocess_fn(df)

                if drop_na:
                    df = df.dropna()

                print(f"After preprocessing the data has {df.shape[0]} entries with {df.shape[1]} features each.")

                return df

            model = preprocessing_model.PreprocessingModel(
                model, preprocess_drop_na_fn, postprocess_fn
            )

            return model

        except (ModuleNotFoundError, FileNotFoundError) as e:
            raise automodel.ParsingError(
                "UNABLE TO READ THE MODEL FILE.:\n" + e.msg
            )

    else:
        print("Parsing model from txt file (was unable to read it as python module containing numpyro code)")
        return _load_cli_dpvi_automodel(
            args.model_path,
            args.k,
            args.clipping_threshold,
            args.num_epochs,
            args.sampling_ratio,
            args.drop_na
        )
