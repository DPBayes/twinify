import jax.numpy as np
from jax.config import config

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from dppp.minibatch import q_to_batch_size, batch_size_to_q
from dppp.dputil import approximate_sigma_remove_relation
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal, AutoContinuousELBO

import fourier_accountant

from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

import pandas as pd

import importlib.util

import jax, argparse, pickle
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Script for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='path to target data')
parser.add_argument('model_path', type=str, help='path to model')
parser.add_argument("output_path", type=str, help="path to outputs (synthetic data and model)")
parser.add_argument("--epsilon", default=1., type=float, help="target privacy parameter")
parser.add_argument("--seed", default=0, type=int, help="PRNG seed used in model fitting")
parser.add_argument("--k", default=5, type=int, help="mixture components in fit")
parser.add_argument("--num_epochs", "-e", default=100, type=int, help="number of epochs")
parser.add_argument("--sampling_ratio", "-q", default=0.01, type=float, help="subsampling ratio for DP-SGD")
parser.add_argument("--num_synthetic", default=1000, type=int, help="amount of synthetic data to generate")

args = parser.parse_args()
print(args)

def main():
    onp.random.seed(args.seed)

    # read data
    df = pd.read_csv(args.data_path)

    # check whether we parse model from txt or whether we have a numpyro module
    try:
        spec = importlib.util.spec_from_file_location("model_module", args.model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model = model_module.model
        model_args_map = model_module.model_args_map
        features = model_module.features
        train_df = df[features].dropna()
        if hasattr(model_module, "feature_maps"):
            feature_maps = model_module.feature_maps
        else: feature_maps = {}
        for name, feature_map in feature_maps.keys():
            train_df[name] = train_df[name].map(feature_maps[name])

    except:
        print("Parsing model from txt file")
        k = args.k
        # read model file
        model_handle = open(args.model_path, 'r')
        model_str = "".join(model_handle.readlines())
        model_handle.close()
        feature_dists, feature_str_dict = automodel.parse_model(model_str, return_str_dict=True)

        # pick features from data according to model file
        train_df = df[list(feature_dists.keys())].dropna()
        # TODO normalize?

        # map features to appropriate values
        feature_maps = {}
        for name, feature_dist in feature_str_dict.items():
            if feature_dist in ["Categorical", "Bernoulli"]:
                feature_maps[name] = {val : iterator for iterator, val in enumerate(onp.unique(train_df[name]))}
                train_df[name] = train_df[name].map(feature_maps[name])


        # shape look-up
        shapes = {name : (k,) if dist!="Categorical" else (k, len(onp.unique(df[name].dropna()))) \
                for name, dist in feature_str_dict.items()}
        feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)

        # build model
        prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
        model = automodel.make_model(feature_dists_and_shapes, prior_dists, k)
        model_args_map = automodel.model_args_map

    # build variational guide for optimization
    guide = AutoDiagonalNormal(make_observed_model(model, model_args_map))

    # pick features from data according to model file
    num_data = train_df.shape[0]
    print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))

    # compute DP values
    target_delta = 1. / num_data
    num_compositions = int(args.num_epochs / args.sampling_ratio)
    dp_sigma, epsilon, _ = approximate_sigma_remove_relation(
        args.epsilon, target_delta, args.sampling_ratio, num_compositions
    )
    batch_size = q_to_batch_size(args.sampling_ratio, num_data)
    sigma_per_sample = dp_sigma / q_to_batch_size(args.sampling_ratio, num_data)
    print("Will apply noise with variance {:.2f} (~ {:.2f} per element in batch) to achieve privacy epsilon "\
        "of {:.3f} (for delta {:.2e}) ".format(dp_sigma, sigma_per_sample, epsilon, target_delta))

    # TODO: warn for high noise? but when is it too high? what is a good heuristic?


    # learn posterior distributions
    posterior_params = train_model(
        jax.random.PRNGKey(args.seed),
        model, automodel.model_args_map, guide, None,
        AutoContinuousELBO(),
        train_df.to_numpy(),
        batch_size=int(args.sampling_ratio*len(train_df)),
        num_epochs=args.num_epochs,
        dp_scale=dp_sigma
    )

    # sample synthetic data from posterior predictive distribution
    posterior_samples = sample_multi_posterior_predictive(jax.random.PRNGKey(args.seed + 1),\
            args.num_synthetic, model, (1,), guide, (), posterior_params)
    syn_data = posterior_samples['x']

    # save results
    syn_df = pd.DataFrame(syn_data, columns = train_df.columns)
    for name, forward_map in feature_maps.items():
        inverse_map = {value: key for key, value in forward_map.items()}
        syn_df[name] = syn_df[name].map(inverse_map)
    syn_df.to_csv("{}.csv".format(args.output_path))
    pickle.dump(posterior_params, open("{}.p".format(args.output_path), "wb"))

    # TODO
    # illustrate

if __name__ == "__main__":
        main()


