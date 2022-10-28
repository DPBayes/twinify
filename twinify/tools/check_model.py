#!/usr/bin/env python

# Copyright 2020, 2021 twinify Developers and their Assignees

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

from jax.config import config
config.update("jax_enable_x64", True)
import jax

import pandas as pd
import numpy as np
from numpyro.handlers import trace, seed
from numpyro.infer import Predictive
import d3p.random
from twinify.infer import train_model_no_dp
from twinify.model_loading import ModelException, load_custom_numpyro_model
from twinify.sampling import sample_synthetic_data, reshape_and_postprocess_synthetic_data

def setup_argument_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("--drop_na", default=False, action='store_true', help="Remove missing values from data.")
    parser.add_argument("--full_traceback", default=False, action='store_true', help="Print a full traceback when errors occur, instead of filtering for custom model code.")

    # we mirror all arguments of the twinify main script here as models may now use any of these
    # in the model_factory method. They are not used by the check-model script.
    parser.add_argument("--epsilon", default=1., type=float, help="[UNUSED] Target multiplicative privacy parameter epsilon.")
    parser.add_argument("--delta", default=None, type=float, help="[UNUSED] Target additive privacy parameter delta.")
    parser.add_argument("--clipping_threshold", default=1., type=float, help="[UNUSED] Clipping threshold for DP-SGD.")
    parser.add_argument("--seed", default=None, type=int, help="[UNUSED] PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
    parser.add_argument("--k", default=50, type=int, help="[UNUSED] Mixture components in fit (for automatic modelling only).")
    parser.add_argument("--num_epochs", "-e", default=200, type=int, help="[UNUSED] Number of training epochs.")
    parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="[UNUSED] Subsampling ratio for DP-SGD.")
    parser.add_argument("--num_synthetic", "--n", default=None, type=int, help="Amount of synthetic data to generate in total. By default as many as input data.")
    parser.add_argument("--num_synthetic_records_per_parameter_sample", "--m", default=1, type=int, help="Amount of synthetic samples to sample per parameter value drawn from the learned parameter posterior.")
    parser.add_argument("--visualize", default="both", choices=["none", "store", "popup", "both"], help="[UNUSED] Options for visualizing the sampled synthetic data. none: no visualization, store: plots are saved to the filesystem, popup: plots are displayed in popup windows, both: plots are saved to the filesystem and displayed")
    parser.add_argument("--no-privacy", default=False, action='store_true', help="[UNUSED] Turn off all privacy features. Intended FOR DEBUGGING ONLY")



def main(args: argparse.Namespace, unknown_args: Iterable[str]) -> int:
    # read data
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        exit(1)

    args = argparse.Namespace(**vars(args), output_path='')

    train_df = df.copy()
    if args.drop_na:
        train_df = train_df.dropna()
    num_data = 100

    try:
        # loading the model
        if args.model_path[-3:] == '.py':
            try:
                model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df)
            except (ModuleNotFoundError, FileNotFoundError) as e:
                print("#### COULD NOT FIND THE MODEL FILE ####")
                print(e)
                exit(1)
        else:
            print("#### loading txt file model currently not supported ####")
            exit(2)

        print("Extracting relevant features from data (using preprocess)")
        zeroed_train_data, _, feature_names = preprocess_fn(train_df.iloc[:2])
        zeroed_train_data = tuple(np.zeros_like(df) for df in zeroed_train_data)

        print("Sampling from prior distribution (using model, guide)")
        # We use Preditive with model to sample from the prior predictive distribution. Since this does not inolve guide,
        # Predictive has no clue about which of the samples are for observations and which are for parameter values.
        # Since we expect postprocess_fn to deal only with observations, we trace through guide to identify
        # parameter sample sites and filter those out. (To invoke guide we need a small batch of data, for which we
        # use whatever preprocess_fn returned to get the right shapes, but zero it out to prevent information leakage).
        try:
            prior_samples = Predictive(model, num_samples = num_data)(jax.random.PRNGKey(0))
        except Exception as e: raise ModelException("Error while obtaining prior samples from model", base_exception=e)
        try:
            parameter_sites = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(*zeroed_train_data)
        except Exception as e: raise ModelException("Error while determining the sampling sites of parameter priors")
        parameter_sites = parameter_sites.keys()
        prior_samples = {site: np.asarray(samples.squeeze(1)) for site, samples in prior_samples.items() if site not in parameter_sites}

        print("Transforming prior samples to output domain to obtain dummy data (using postprocess)")
        _, syn_prior_encoded = postprocess_fn(prior_samples, df, feature_names)

        print("Preprocessing dummy data (using preprocess)")
        train_data, num_train_data, feature_names = preprocess_fn(syn_prior_encoded)

        assert isinstance(train_data, tuple)
        assert num_train_data == num_data # TODO: maybe not?

        print("Inferring model parameters (using model, guide)")
        try:
            posterior_params, _ = train_model_no_dp(d3p.random.PRNGKey(0),
                model, guide,
                train_data,
                batch_size = num_train_data//2,
                num_data = num_train_data,
                num_epochs = 3,
                silent = True
            )
        except Exception as e:
            raise ModelException("Error while performing inference", base_exception=e)

        print("Sampling from posterior distribution (using model, guide)")
        try:
            posterior_samples = sample_synthetic_data(model, guide, posterior_params, jax.random.PRNGKey(0), num_train_data, num_train_data)
        except Exception as e:
            raise ModelException("Error while obtaining posterior samples from model", base_exception=e)
        print("Postprocessing (using postprocess)")
        conditioned_postprocess_fn = lambda samples: postprocess_fn(samples, df, feature_names)
        reshape_and_postprocess_synthetic_data(
            posterior_samples, conditioned_postprocess_fn, separate_output=True
        )

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
        main(parser.parse_args())
    )
