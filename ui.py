from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal

import fourier_accountant

from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

import pandas as pd

import jax, argparse, pickle


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

def main():
    onp.random.seed(args.seed)

    k = args.k
    # read data
    df = pd.read_csv(args.data_path)

    # read model file
    model_handle = open(args.model_path, 'r')
    model_str = "".join(model_handle.readlines())
    model_handle.close()
    features = automodel.parse_model(model_str)
    feature_names = [feature.name for feature in features]

    # pick features from data according to model file
    train_df = df[feature_names].dropna()
    print("After removing missing values, the data has {} entries with {} features".format(*train_df.shape))

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

    # data preprocessing: determines number of categories for Categorical
    #   distribution and maps categorical values in the data to ints
    for feature in features:
        train_df = feature.preprocess_data(train_df)

    # TODO normalize?

    # build model
    model = automodel.make_model(features, k)

    # build variational guide for optimization
    guide = AutoDiagonalNormal(make_observed_model(model, automodel.model_args_map))

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

    # postprocess: if preprocessing involved data mapping, it is mapped back here
    #   so that the synthetic twin looks like the original data
    for feature in features:
        syn_df = feature.postprocess_data(syn_df)

    syn_df.to_csv("{}.csv".format(args.output_path))
    pickle.dump(posterior_params, open("{}.p".format(args.output_path), "wb"))

    # TODO
    # illustrate

if __name__ == "__main__":
        main()


