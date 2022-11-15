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

from numpyro.infer.autoguide import AutoDiagonalNormal

import twinify.dpvi.modelling.automodel as automodel
from twinify.model_loading import ModelException, load_custom_numpyro_model
from twinify.dpvi.sampling import reshape_and_postprocess_synthetic_data

import numpy as np

import pandas as pd

import argparse
import d3p.random

from twinify.dpvi import DPVIModel, DPVIResult, InferenceException

from twinify import __version__

parser = argparse.ArgumentParser(description='Twinify: Program for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
parser.add_argument("output_path", type=str, help="Path prefix to outputs (synthetic data, model and visuliatzion plots).")
parser.add_argument("--epsilon", default=1., type=float, help="Target multiplicative privacy parameter epsilon.")
parser.add_argument("--delta", default=None, type=float, help="Target additive privacy parameter delta.")
parser.add_argument("--clipping_threshold", default=1., type=float, help="Clipping threshold for DP-SGD.")
parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a random value.")
parser.add_argument("--k", default=50, type=int, help="Mixture components in fit (for automatic modelling only).")
parser.add_argument("--num_epochs", "-e", default=200, type=int, help="Number of training epochs.")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="Subsampling ratio for DP-SGD.")
parser.add_argument("--num_synthetic", "--n", default=None, type=int, help="Amount of synthetic data to generate in total. By default as many as input data.")
parser.add_argument("--num_synthetic_records_per_parameter_sample", "--m", default=1, type=int, help="Amount of synthetic samples to sample per parameter value drawn from the learned parameter posterior.")
parser.add_argument("--drop_na", default=0, type=int, help="Remove missing values from data (yes=1)")
parser.add_argument("--visualize", default="both", choices=["none", "store", "popup", "both"], help="Options for visualizing the sampled synthetic data. none: no visualization, store: plots are saved to the filesystem, popup: plots are displayed in popup windows, both: plots are saved to the filesystem and displayed")
parser.add_argument("--no-privacy", default=False, action='store_true', help="Turn off all privacy features. Intended FOR DEBUGGING ONLY!")
parser.add_argument("--separate_output", default=False, action='store_true', help="Store synthetic data in separate files per parameter sample.")
parser.add_argument("--version", action='version', version=__version__)

def initialize_rngs(seed):
    master_rng = d3p.random.PRNGKey(seed)
    print(f"RNG seed: {seed}")

    inference_rng, sampling_rng, numpyro_seed = d3p.random.split(master_rng, 3)
    sampling_rng = d3p.random.convert_to_jax_rng_key(sampling_rng)

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
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        return 1
    print("Loaded data set has {} rows (entries) and {} columns (features).".format(*df.shape))
    num_data = len(df)

    try:
    # check whether we parse model from txt or whether we have a numpyro module
        if args.model_path[-3:] == '.py':

            train_df = df.copy()
            if args.drop_na:
                train_df = train_df.dropna()

            try:
                model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path, args, unknown_args, train_df)
            except (ModuleNotFoundError, FileNotFoundError) as e:
                print("#### COULD NOT FIND THE MODEL FILE ####")
                print(e)
                return 1

            train_data, num_data, feature_names = preprocess_fn(train_df)
        else:
            print("Parsing model from txt file (was unable to read it as python module containing numpyro code)")
            k = args.k
            # read model file
            with open(args.model_path, 'r') as model_handle:
                model_str = "".join(model_handle.readlines())
            features = automodel.parse_model(model_str)
            feature_names = [feature.name for feature in features]

            # pick features from data according to model file
            missing_features = set(feature_names).difference(df.columns)
            if missing_features:
                raise automodel.ParsingError(
                    "The model specifies features that are not present in the data:\n{}".format(
                        ", ".join(missing_features)
                    )
                )

            df = df.loc[:, feature_names]

            train_df = df.copy() # TODO: this duplicates code with the other branch but cannot currently pull it out because we are manipulating df above
            if args.drop_na:
                train_df = train_df.dropna()

            # data preprocessing: determines number of categories for Categorical
            #   distribution and maps categorical values in the data to ints
            for feature in features:
                train_df = feature.preprocess_data(train_df)

            # build model
            model = automodel.make_model(features, k)

            # build variational guide for optimization
            guide = AutoDiagonalNormal(model)

            # postprocessing for automodel
            postprocess_fn = automodel.postprocess_function_factory(features)
            num_data = train_df.shape[0]
            train_data = (train_df,)

        assert isinstance(train_data, tuple)
        if len(train_data) == 1:
            print("After preprocessing, the data has {} entries with {} features each.".format(*train_data[0].shape))
        else:
            print("After preprocessing, the data was split into {} splits:".format(len(train_data)))
            for i, x in enumerate(train_data):
                print("\tSplit {} has {} entries with {} features each.".format(i, x.shape[0], 1 if x.ndim == 1 else x.shape[1]))

        # compute DP values
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

        output_sampling_sites = ["xs"] # TODO: need to sort these out
        dpvi_model = DPVIModel(model, output_sampling_sites, guide)

        try:
            dpvi_result: DPVIResult = dpvi_model.fit(
                train_data,
                inference_rng,
                args.epsilon,
                target_delta,
                args.clipping_threshold,
                args.num_epochs,
                args.sampling_ratio
            )
        except (InferenceException, FloatingPointError):
            print("################################## ERROR ##################################")
            print("!!!!! The inference procedure encountered a NaN value (not a number). !!!!!")
            print("This means the model has major difficulties in capturing the data and is")
            print("likely to happen when the dataset is very small and/or sparse.")
            print("Try adapting (simplifying) the model.")
            print("Aborting...")
            return 2

        dp_sigma = dpvi_result.privacy_level.dp_noise
        act_epsilon = dpvi_result.privacy_level.epsilon
        act_delta = dpvi_result.privacy_level.delta
        sigma_per_sample = dp_sigma / DPVIModel.batch_size_for_subsample_ratio(args.sampling_ratio, num_data)
        print("Will apply noise with std deviation {:.2f} (~ {:.2f} per element in batch) to achieve privacy epsilon "\
            "of {:.3f} (for delta {:.2e}) ".format(dp_sigma, sigma_per_sample, act_epsilon, act_delta))
        # TODO: warn for high noise? but when is it too high? what is a good heuristic?

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

        posterior_samples = dpvi_result.generate_extended(
            sampling_rng,
            args.num_synthetic_records_per_parameter_sample,
            num_parameter_samples,
            single_dataframe=not args.separate_output
        )

        # postprocess: so that the synthetic twin looks like the original data
        #   - extract samples from the posterior_samples dictionary and construct pd.DataFrame
        #   - if preprocessing involved data mapping, it is mapped back here
        conditioned_postprocess_fn = lambda posterior_samples: postprocess_fn(posterior_samples, df, feature_names)
        for i, (syn_df, encoded_syn_df) in enumerate(reshape_and_postprocess_synthetic_data(
            posterior_samples, conditioned_postprocess_fn, args.separate_output
        )):
            if args.separate_output:
                filename = f"{args.output_path}.{i}.csv"
            else:
                filename = f"{args.output_path}.csv"
            encoded_syn_df.to_csv(filename, index=False)

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
