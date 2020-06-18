from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from dppp.minibatch import q_to_batch_size, batch_size_to_q
from dppp.dputil import approximate_sigma_remove_relation
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal
from numpyro.infer import Predictive

import fourier_accountant

from twinify.infer import train_model, train_model_no_dp, InferenceException
import twinify.automodel as automodel

import numpy as onp

import pandas as pd

import importlib.util
import traceback

import jax, argparse, pickle
import secrets


parser = argparse.ArgumentParser(description='Script for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='path to target data')
parser.add_argument('model_path', type=str, help='path to model')
parser.add_argument("output_path", type=str, help="path to outputs (synthetic data and model)")
parser.add_argument("--epsilon", default=1., type=float, help="target privacy parameter")
parser.add_argument("--seed", default=None, type=int, help="PRNG seed used in model fitting. If not set, will be securely initialized to a unique value.")
parser.add_argument("--k", default=5, type=int, help="mixture components in fit")
parser.add_argument("--num_epochs", "-e", default=100, type=int, help="number of epochs")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="subsampling ratio for DP-SGD")
parser.add_argument("--num_synthetic", default=1000, type=int, help="amount of synthetic data to generate")
parser.add_argument("--drop_na", default=0, type=int, help="remove missing values from data (yes=1)")
parser.add_argument("--clipping_threshold", default=1., type=float, help="clipping threshold")

def initialize_rngs(seed):
    if seed is None:
        seed = secrets.randbelow(2**32)
    print("RNG seed: {}".format(seed))
    master_rng = jax.random.PRNGKey(seed)
    onp.random.seed(seed)
    return jax.random.split(master_rng, 2)

class ParsingError(Exception):
    pass

def main(args):
    # read data
    df = pd.read_csv(args.data_path)

    # check whether we parse model from txt or whether we have a numpyro module
    try:
        if args.model_path[-3:] == '.py':
            spec = importlib.util.spec_from_file_location("model_module", args.model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            model = model_module.model
            # feature_names = model_module.features

            # features = automodel.parse_model_fn(model, feature_names)

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
            print("Parsing model from txt file (was unable to read it python module containing numpyro code)")
            k = args.k
            # read model file
            model_handle = open(args.model_path, 'r')
            model_str = "".join(model_handle.readlines())
            model_handle.close()
            features = automodel.parse_model(model_str)
            feature_names = [feature.name for feature in features]

            # pick features from data according to model file
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

    except Exception as e:
        print("#### FAILED TO PARSE THE MODEL SPECIFICATION ####")
        print("Here's the technical error description:")
        print(e)
        traceback.print_tb(e.__traceback__)
        print("Aborting...")
        exit(3)

    # pick features from data according to model file
    num_data = train_df.shape[0]
    if args.drop_na:
        print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))
    else:
        print("The data has {} entries with {} features".format(*train_df.shape))

    # compute DP values
    target_delta = 1. / num_data
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
        posterior_params = train_model(
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

    predictive_model = lambda: model(None)
    posterior_samples = Predictive(predictive_model, guide=guide, params=posterior_params, num_samples=args.num_synthetic).get_samples(sampling_rng)

    # sample synthetic data from posterior predictive distribution
    # posterior_samples = sample_multi_posterior_predictive(sampling_rng,
    #         args.num_synthetic, model, (None,), guide, (), posterior_params)
    syn_data = posterior_samples['x']

    # save results
    syn_df = pd.DataFrame(syn_data, columns = train_df.columns)

    # postprocess: if preprocessing involved data mapping, it is mapped back here
    #   so that the synthetic twin looks like the original data
    if postprocess_fn:
        syn_df = postprocess_fn(syn_df)

    syn_df.to_csv("{}.csv".format(args.output_path), index=False)
    pickle.dump(posterior_params, open("{}.p".format(args.output_path), "wb"))

    # TODO
    # illustrate

if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    main(args)


