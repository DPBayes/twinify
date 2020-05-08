import jax.numpy as np
from jax.config import config

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal

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
parser.add_argument("--dp_sigma", default=1., type=float, help="sigma value for noise in DP SVI")
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
        print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))

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

    # compute DP values
    target_delta = 1. / train_df.shape[0]
    num_compositions = int(args.num_epochs / args.sampling_ratio)
    L = 20.
    try:
        epsilon = fourier_accountant.get_epsilon_R(
            target_delta, args.dp_sigma, args.sampling_ratio, num_compositions, L=L
        )
        print("With the chosen parameters you obtain a privacy epsilon of {:.3f} (for delta {:.2e}).".format(epsilon, target_delta))
    except ValueError:
        epsilon = np.inf
        print("With the chosen parameters the privacy epsilon exceeds {} (for delta {:.2e}).".format(L, target_delta))

    if (epsilon > 2.):
        print("!!! THIS IS BAD !!!")
        print("NOTE: As a rule of thumb, epsilon values should not exceed 2!")
        print("      You should consider increasing the privacy noise (dp_sigma parameter).")
        response = input("Do you want to proceed DESPITE NOT HAVING ADEQUATE PRIVACY GUARANTEES? (yes, no) ")
        if response.lower() != "yes":
            print("Terminating")
            exit(1)
        print("Continuing (you have been warned)...")



    # learn posterior distributions
    posterior_params = train_model(
        jax.random.PRNGKey(args.seed),
        model, automodel.model_args_map, guide, None,
        train_df.to_numpy(),
        batch_size=int(args.sampling_ratio*len(train_df)),
        num_epochs=args.num_epochs,
        dp_scale=args.dp_sigma
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


