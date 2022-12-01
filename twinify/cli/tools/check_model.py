#!/usr/bin/env python

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

"""
Twinify custom model checking mode script.
"""

import argparse
from typing import Iterable

import jax

import pandas as pd
import numpy as np
import numpyro
from numpyro.handlers import trace, seed
from numpyro.infer import Predictive
from twinify.cli.dpvi_loader import load_cli_dpvi
from twinify import DataDescription
from twinify.base import InferenceModel
from twinify.cli.preprocessing_model import PreprocessingModel
from twinify.dpvi.dpvi_model import DPVIModel, DPVIResult
from twinify.cli.dpvi_numpyro_model_loading import ModelException
import d3p.random

from twinify.dpvi.sampling import sample_synthetic_data

def setup_argument_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.py).')
    parser.add_argument("--drop_na", default=False, action='store_true', help="Remove missing values from data.")
    parser.add_argument("--full_traceback", default=False, action='store_true', help="Print a full traceback when errors occur, instead of filtering for custom model code.")

    # we mirror all arguments of the twinify main script here as models may now use any of these
    # in the model_factory method. They are not used by the check-model script.
    parser.add_argument("--epsilon", default=1., type=float, help="[UNUSED] Target multiplicative privacy parameter epsilon.")
    parser.add_argument("--delta", default=None, type=float, help="[UNUSED] Target additive privacy parameter delta.")
    parser.add_argument("--clipping_threshold", default=1., type=float, help="[UNUSED] Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="[UNUSED] PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="[UNUSED] Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=2, type=int, help="[UNUSED] Number of training epochs.")
    parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="[UNUSED] Subsampling ratio for DP-SGD.")
    parser.add_argument("--num_synthetic", "--n", default=None, type=int, help="Amount of synthetic data to generate in total. By default as many as input data.")
    parser.add_argument("--num_synthetic_records_per_parameter_sample", "--m", default=1, type=int, help="Amount of synthetic samples to sample per parameter value drawn from the learned parameter posterior.")



def main(args: argparse.Namespace, unknown_args: Iterable[str]) -> int:
    # read data
    try:
        train_df = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        exit(1)

    args.output_path = ''
    args.num_epochs = 2

    data_description = DataDescription.from_dataframe(train_df)

    num_data = 100
    train_df = train_df.iloc[:num_data]


    try:
        model: PreprocessingModel = load_cli_dpvi(args, unknown_args, data_description)

        preprocessed_train_df = model.preprocess(train_df)
        data_description = DataDescription.from_dataframe(preprocessed_train_df)

        dpvi_model: DPVIModel = model._base_model

        print("Sampling from prior distribution (using model, guide)")
        # We use Preditive with model to sample from the prior predictive distribution. Since this does not inolve guide,
        # Predictive has no clue about which of the samples are for observations and which are for parameter values.
        # Since we expect postprocess_fn to deal only with observations, we trace through guide to identify
        # parameter sample sites and filter those out. (To invoke guide we need a small batch of data, for which we
        # use whatever preprocess_fn returned to get the right shapes, but zero it out to prevent information leakage).
        try:
            prior_samples = Predictive(DPVIResult._mark_model_outputs(dpvi_model._model), num_samples = num_data)(jax.random.PRNGKey(0))
        except Exception as e: raise ModelException("Error while obtaining prior samples from model", base_exception=e)

        prior_samples = prior_samples[DPVIResult._twinify_model_output_site].squeeze(1)

        print("Transforming prior samples to output domain to obtain dummy data (using postprocess)")
        # prior_samples = pd.DataFrame({site: np.asarray(samples.squeeze(1)) for site, samples in prior_samples.items() if site not in parameter_sites})
        prior_samples = pd.DataFrame(prior_samples, columns=data_description.columns)
        prior_samples = data_description.map_to_categorical(prior_samples)
        if model.postprocess_fn is not None:
            prior_samples = model.postprocess_fn(prior_samples)

        assert len(prior_samples) == num_data # TODO: maybe not?

        print("Preprocessing dummy data (using preprocess) and inferring model parameters (using model, guide)")
        try:
            result = model.fit(
                prior_samples,
                d3p.random.PRNGKey(13),
                epsilon=10.,
                delta=0.1
            )
        except Exception as e:
            raise ModelException("Error while performing inference", base_exception=e)

        print("Sampling from posterior distribution (using model, guide) and postprocessing (using postprocess)")
        try:
            posterior_samples = result.generate(d3p.random.PRNGKey(0), num_data, num_data)
        except Exception as e:
            raise ModelException("Error while obtaining posterior samples from model", base_exception=e)

        print("Everything okay!")
        return 0

    except ModelException as e:
        if args.full_traceback:
            print(e)
        else:
            print(e.format_message(args.model_path))
    except AssertionError as e:
        raise e
    except Exception as e:
        print("#### AN UNCATEGORISED ERROR OCCURRED ####")
        raise e
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Twinify: Program for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
    setup_argument_parser(parser)
    exit(
        main(parser.parse_known_args())
    )
