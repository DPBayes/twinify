#!/usr/bin/env python

# Copyright 2020 twinify Developers and their Assignees

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
Copy of Twinify main script using standard SVI (NO DP!) for comparison of quality of results.
"""

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np

from dppp.minibatch import q_to_batch_size, batch_size_to_q
from dppp.dputil import approximate_sigma_remove_relation
from numpyro.handlers import seed
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import Predictive

import fourier_accountant

from twinify.infer import train_model, train_model_no_dp, InferenceException
import twinify.dpvi.modelling.automodel as automodel

import numpy as onp

import pandas as pd

import importlib.util
import traceback

import jax, argparse, pickle
import secrets

from twinify.illustrate import plot_missing_values, plot_margins, plot_covariance_heatmap
import matplotlib.pyplot as plt

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
parser.add_argument("--num_synthetic", default=None, type=int, help="Amount of synthetic data to generate. By default as many as input data.")
parser.add_argument("--drop_na", default=0, type=int, help="Remove missing values from data (yes=1)")
parser.add_argument("--visualize", default="both", choices=["none", "store", "popup", "both"], help="Options for visualizing the sampled synthetic data. none: no visualization, store: plots are saved to the filesystem, popup: plots are displayed in popup windows, both: plots are saved to the filesystem and displayed")

def initialize_rngs(seed):
    if seed is None:
        seed = secrets.randbelow(2**32)
    print("RNG seed: {}".format(seed))
    master_rng = jax.random.PRNGKey(seed)
    onp.random.seed(seed)
    return jax.random.split(master_rng, 2)

def main():
    args = parser.parse_args()
    print(args)

    # read data
    df = pd.read_csv(args.data_path)

    # check whether we parse model from txt or whether we have a numpyro module
    try:
        if args.model_path[-3:] == '.py':
            spec = importlib.util.spec_from_file_location("model_module", args.model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            model = model_module.model

            train_df = df
            if args.drop_na:
                train_df = train_df.dropna()

            ## AUTOMATIC PREPROCESSING CURRENTLY UNAVAILABLE
            # data preprocessing: determines number of categories for Categorical
            #   distribution and maps categorical values in the data to ints
            # for feature in features:
            #     train_df = feature.preprocess_data(train_df)

            ## ALTERNATIVE
            # we do allow the user to specify a preprocess/postprocess function pair
            # in the numpyro model file
            try: preprocess_fn = model_module.preprocess
            except: preprocess_fn = None
            if preprocess_fn:
                train_df = preprocess_fn(train_df)

            try: postprocess_fn = model_module.postprocess
            except: postprocess_fn = None

            try: guide = model_module.guide
            except: guide = AutoDiagonalNormal(model)

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

            train_df = df.loc[:, feature_names]
            if args.drop_na:
                train_df = train_df.dropna()

            # TODO normalize?

            # data preprocessing: determines number of categories for Categorical
            #   distribution and maps categorical values in the data to ints
            for feature in features:
                train_df = feature.preprocess_data(train_df)

            # build model
            model = automodel.make_model(features, k)

            # build variational guide for optimization
            guide = AutoDiagonalNormal(model)

            # postprocessing for automodel
            def postprocess_fn(syn_df):
                for feature in features:
                    syn_df = feature.postprocess_data(syn_df)
                return syn_df

    except Exception as e: # handling errors in py-file parsing
        print("\n#### FAILED TO PARSE THE MODEL SPECIFICATION ####")
        print("Here's the technical error description:")
        print(e)
        traceback.print_tb(e.__traceback__)
        print("\nAborting...")
        exit(3)

    # pick features from data according to model file
    num_data = train_df.shape[0]
    if args.drop_na:
        print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))
    else:
        print("The data has {} entries with {} features".format(*train_df.shape))

    # compute DP values
    target_delta = args.delta
    if target_delta is None:
        target_delta = 1. / num_data
    if target_delta * num_data > 1.:
        print("!!!!! WARNING !!!!! The given value for privacy parameter delta ({:1.3e}) exceeds 1/(number of data) ({:1.3e}),\n" \
            "which the maximum value that is usually considered safe!".format(
                target_delta, 1. / num_data
            ))
        x = input("Continue? (type YES ): ")
        if x != "YES":
            print("Aborting...")
            exit(4)
        print("Continuing... (YOU HAVE BEEN WARNED!)")

    num_compositions = int(args.num_epochs / args.sampling_ratio)
    dp_sigma, epsilon, _ = approximate_sigma_remove_relation(
        args.epsilon, target_delta, args.sampling_ratio, num_compositions
    )
    batch_size = q_to_batch_size(args.sampling_ratio, num_data)
    sigma_per_sample = dp_sigma / q_to_batch_size(args.sampling_ratio, num_data)
    print("Will apply noise with std deviation {:.2f} (~ {:.2f} per element in batch) to achieve privacy epsilon "\
        "of {:.3f} (for delta {:.2e}) ".format(dp_sigma, sigma_per_sample, epsilon, target_delta))

    # TODO: warn for high noise? but when is it too high? what is a good heuristic?

    inference_rng, sampling_rng = initialize_rngs(args.seed)

    # learn posterior distributions
    try:
        posterior_params = train_model_no_dp(
            inference_rng,
            model, guide,
            train_df.to_numpy(),
            batch_size=int(args.sampling_ratio*len(train_df)),
            num_epochs=args.num_epochs,
            dp_scale=dp_sigma,
            clipping_threshold=args.clipping_threshold
        )
    except (InferenceException, FloatingPointError):
        print("################################## ERROR ##################################")
        print("!!!!! The inference procedure encountered a NaN value (not a number). !!!!!")
        print("This means the model has major difficulties in capturing the data and is")
        print("likely to happen when the dataset is very small and/or sparse.")
        print("Try adapting (simplifying) the model.")
        print("Aborting...")
        exit(2)

    num_synthetic = args.num_synthetic
    if num_synthetic is None:
        num_synthetic = train_df.shape[0]

    predictive_model = lambda: model(None)
    posterior_samples = Predictive(
        predictive_model, guide=guide, params=posterior_params,
        num_samples=num_synthetic
    ).get_samples(sampling_rng)

    # sample synthetic data from posterior predictive distribution
    # posterior_samples = sample_multi_posterior_predictive(sampling_rng,
    #         args.num_synthetic, model, (None,), guide, (), posterior_params)
    syn_data = posterior_samples['x']

    # save results
    syn_df = pd.DataFrame(syn_data, columns = train_df.columns)

    # postprocess: if preprocessing involved data mapping, it is mapped back here
    #   so that the synthetic twin looks like the original data
    encoded_syn_df = syn_df.copy()
    if postprocess_fn:
        encoded_syn_df = postprocess_fn(encoded_syn_df)

    encoded_syn_df.to_csv("{}.csv".format(args.output_path), index=False)
    pickle.dump(posterior_params, open("{}.p".format(args.output_path), "wb"))

    ## illustrate results
    if args.visualize != 'none':
        show_popups = args.visualize in ('popup', 'both')
        save_plots = args.visualize in ('store', 'both')
        # Missing value rate
        if not args.drop_na:
            missing_value_fig = plot_missing_values(syn_df, train_df, show=show_popups)
            if save_plots:
                missing_value_fig.savefig(args.output_path + "_missing_value_plots.svg")
        # Marginal violins
        margin_fig = plot_margins(syn_df, train_df, show=show_popups)
        # Covariance matrices
        cov_fig = plot_covariance_heatmap(syn_df, train_df, show=show_popups)
        if save_plots:
            margin_fig.savefig(args.output_path + "_marginal_plots.svg")
            cov_fig.savefig(args.output_path + "_correlation_plots.svg")
        if show_popups:
            plt.show()

if __name__ == "__main__":
    main()
