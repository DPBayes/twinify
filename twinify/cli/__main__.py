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
Twinify main script.
"""

from jax.config import config
config.update("jax_enable_x64", True)

from twinify.cli.model_loading import ModelException

import numpy as np
import pandas as pd
from typing import Optional

import argparse
import d3p.random

from twinify.base import InferenceModel, InferenceResult
from twinify.dpvi import InferenceException
from twinify.cli.dpvi_loader import load_cli_dpvi
from twinify.cli.napsu_loader import load_cli_napsu
from twinify import DataDescription

from twinify import __version__

parser = argparse.ArgumentParser(description='Twinify: Program for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
subparsers = parser.add_subparsers(title='inference_method')
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
parser.add_argument("output_path", type=str, help="Path prefix to outputs (synthetic data, model and visuliatzion plots).")
parser.add_argument("--epsilon", default=1., type=float, help="Target multiplicative privacy parameter epsilon.")
parser.add_argument("--delta", default=None, type=float, help="Target additive privacy parameter delta.")
parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
parser.add_argument("--num_synthetic", "--n", default=None, type=int, help="Amount of synthetic data to generate in total. By default as many as input data.")
parser.add_argument("--num_synthetic_records_per_parameter_sample", "--m", default=1, type=int, help="Amount of synthetic samples to sample per parameter value drawn from the learned parameter posterior.")
parser.add_argument("--drop_na", default=0, type=int, help="Remove missing values from data (yes=1)")
parser.add_argument("--separate_output", default=False, action='store_true', help="Store synthetic data in separate files per parameter sample.")
parser.add_argument("--version", action='version', version=__version__)

vi_parser = subparsers.add_parser("vi")
vi_parser.add_argument("--clipping_threshold", default=1., type=float, help="Clipping threshold for DP-SGD.")
vi_parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
vi_parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
vi_parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
vi_parser.add_argument("--no-privacy", default=False, action='store_true', help="Turn off all privacy features, i.e., run privacy-agnostic variational inference. Intended FOR DEBUGGING ONLY!")
vi_parser.set_defaults(load_fn=load_cli_dpvi)

napsu_parser = subparsers.add_parser("napsumq")
napsu_parser.set_defaults(load_fn=load_cli_napsu)

def initialize_rngs(seed: Optional[int] = None):
    master_rng = d3p.random.PRNGKey(seed)
    print(f"RNG seed: {seed}")

    inference_rng, sampling_rng, numpyro_seed = d3p.random.split(master_rng, 3)

    numpyro_seed = int(d3p.random.random_bits(numpyro_seed, 32, (1,)))
    np.random.seed(numpyro_seed)

    return inference_rng, sampling_rng

def main():
    args, unknown_args = parser.parse_known_args()
    print(args)
    if unknown_args:
        print(f"Additional received arguments: {unknown_args}")

    # read data
    try:
        train_df = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        return 1
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*train_df.shape))
    num_data = len(train_df)

    data_description = DataDescription.from_dataframe(train_df)

    try:
        model: InferenceModel = args.load_fn(args, unknown_args, data_description)

        if args.drop_na:
            train_df = train_df.dropna()

        print(f"After preprocessing, the data has {train_df.shape[0]} entries with {train_df.shape[1]} features each.")

        # check DP values
        target_delta = args.delta
        if target_delta is None:
            target_delta = 0.1 / num_data
        if target_delta * num_data > 1.:
            print("!!!!! WARNING !!!!! The given value for privacy parameter delta ({:1.3e}) exceeds 1/(number of data) ({:1.3e}),\n" \
                "which the maximum value that is usually considered safe!".format(
                    target_delta, 1. / num_data
                ))
            x = input("Continue? (type YES ): ")
            if x != "YES":
                print("Aborting...")
                return 4
            print("Continuing... (YOU HAVE BEEN WARNED!)")

        inference_rng, sampling_rng = initialize_rngs(args.seed)

        try:
            dpvi_result: InferenceResult = model.fit(
                train_df,
                inference_rng,
                args.epsilon,
                target_delta,
                verbose=True
            )
        except (InferenceException, FloatingPointError):
            print("################################## ERROR ##################################")
            print("!!!!! The inference procedure encountered a NaN value (not a number). !!!!!")
            print("This means the model has major difficulties in capturing the data and is")
            print("likely to happen when the dataset is very small and/or sparse.")
            print("Try adapting (simplifying) the model.")
            print("Aborting...")
            return 2

        # Store learned model parameters
        # TODO: we should have a mode for twinify that allows to rerun the sampling without training, using stored parameters
        dpvi_result.store(f"{args.output_path}.p")
        # TODO: previous twinify would store the entire provided arguments, for reproducability; this is not the case anymore; can we somehow reinstate this?

        # sample synthetic data
        print("Model learning complete; now sampling data!")
        num_synthetic = args.num_synthetic
        if num_synthetic is None:
            num_synthetic = num_data

        num_parameter_samples = int(np.ceil(num_synthetic / args.num_synthetic_records_per_parameter_sample))
        num_synthetic = num_parameter_samples * args.num_synthetic_records_per_parameter_sample
        print(f"Will sample {args.num_synthetic_records_per_parameter_sample} synthetic data records for each of "
              f"{num_parameter_samples} samples from the parameter posterior for a total of {num_synthetic} records.")
        if args.separate_output:
            print("They will be stored in separate data sets for each parameter posterior sample.")
        else:
            print("They will be stored in a single large data set.")

        syn_data = dpvi_result.generate(
            sampling_rng,
            num_parameter_samples,
            args.num_synthetic_records_per_parameter_sample,
            single_dataframe=not args.separate_output
        )

        # store synthetic data
        if isinstance(syn_data, pd.DataFrame):
            syn_data.to_csv(args.output_path + ".csv", index=False)
        else:
            for i, syn_df in enumerate(syn_data):
                syn_df.to_csv(f"{args.output_path}.{i}.csv", index=False)

        ### illustrate results TODO need to adopt new way of handing train_df
        #if args.visualize != 'none':
        #    show_popups = args.visualize in ('popup', 'both')
        #    save_plots = args.visualize in ('store', 'both')
        #    # Missing value rate
        #    if not args.drop_na:
        #        missing_value_fig = plot_missing_values(syn_df, train_df, show=show_popups)
        #        if save_plots:
        #            missing_value_fig.savefig(args.output_path + "_missing_value_plots.svg")
        #    # Marginal violins
        #    margin_fig = plot_margins(syn_df, train_df, show=show_popups)
        #    # Covariance matrices
        #    cov_fig = plot_covariance_heatmap(syn_df, train_df, show=show_popups)
        #    if save_plots:
        #        margin_fig.savefig(args.output_path + "_marginal_plots.svg")
        #        cov_fig.savefig(args.output_path + "_correlation_plots.svg")
        #    if show_popups:
        #        plt.show()
        return 0
    except ModelException as e:
        print(e.format_message(args.model_path))
    except AssertionError as e:
        raise e
    except Exception as e:
        print("#### AN UNCATEGORISED ERROR OCCURRED ####")
        raise e
    return 1

if __name__ == "__main__":
    exit(main())
