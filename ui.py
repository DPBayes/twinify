import jax.numpy as np

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal


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
	feature_dists, feature_str_dict = automodel.parse_model(model_str, return_str_dict=True)

	# pick features from data according to model file
	train_df = df[list(feature_dists.keys())].dropna()

	# map features to appropriate values
	feature_maps = {}
	for name, feature_dist in feature_str_dict.items():
		if feature_dist in ["Categorical", "Bernoulli"]:
			feature_maps[name] = {val : iterator for iterator, val in enumerate(onp.unique(train_df[name]))}
			train_df[name] = train_df[name].map(feature_maps[name])

	# TODO normalize?

	# shape look-up
	shapes = {name : (k,) if dist!="Categorical" else (k, len(onp.unique(df[name].dropna()))) \
			for name, dist in feature_str_dict.items()}
	feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)

	# build model
	prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
	model = automodel.make_model(feature_dists_and_shapes, prior_dists, k)

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
	for name, forward_map in maps.items():
		inverse_map = {value: key for key, value in forward_map.items()}
		syn_df[name] = syn_df[name].map(inverse_map)
	syn_df.to_csv("{}.csv".format(args.output_path))
	pickle.dump(posterior_params, open("{}.p".format(args.output_path), "wb"))

	# TODO
	# report DP cost
	# illustrate

if __name__ == "__main__":
		main()


