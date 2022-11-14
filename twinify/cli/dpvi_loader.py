import argparse
from typing import List, Tuple, Callable
import pandas as pd
import numpyro.infer

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
        subsample_ratio: float
    ) -> twinify.dpvi.DPVIModel:

    with open(model_path, 'r') as f:
        model_str = "".join(f.readlines())

    features = automodel.parse_model(model_str)

    # build model
    model_fn = automodel.make_model(features, mixture_components)

    # build variational guide for optimization
    guide_fn = numpyro.infer.AutoDiagonalNormal(model_fn)

    # postprocessing for automodel
    def preprocess_fn(df: pd.DataFrame) -> pd.DataFrame:

        # pick features from data according to model file
        feature_names = {feature.name for feature in features}
        missing_features = feature_names.difference(df.columns)
        if missing_features:
            raise automodel.ParsingError(
                "The model specifies features that are not present in the data:\n{}".format(
                    ", ".join(missing_features)
                )
            )

        df = df.loc[:, feature_names]

        for feature in features:
            df = feature.preprocess_data(df)

        return df

    postprocess_fn = twinify.dpvi.automodel.postprocess_function_factory(features)

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

            model = preprocessing_model.PreprocessingModel(
                model, preprocess_fn, postprocess_fn
            )

        except (ModuleNotFoundError, FileNotFoundError) as e:
            raise automodel.ParsingError(
                "Unable to read the model file.:\n" + e.msg
            )

    else:
        print("Parsing model from txt file (was unable to read it as python module containing numpyro code)")
        return _load_cli_dpvi_automodel(
            args.model_path,
            args.k,
            args.clipping_threshold,
            args.num_epochs,
            args.sampling_ratio
        )
