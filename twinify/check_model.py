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

from numpyro.infer import Predictive

from twinify.infer import train_model_no_dp, InferenceException
import twinify.automodel as automodel
from twinify.model_loading import load_custom_numpyro_model

import pandas as pd

import jax, argparse, pickle
import secrets

from twinify.illustrate import plot_missing_values, plot_margins, plot_covariance_heatmap
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Twinify: Program for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
parser.add_argument('data_path', type=str, help='Path to input data.')
parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
parser.add_argument("--drop_na", default=False, action='store_true', help="Remove missing values from data.")

def main():
    args = parser.parse_args()

    # read data
    df = pd.read_csv(args.data_path)

    train_df = df.copy()
    if args.drop_na:
        train_df = train_df.dropna()
    num_data = 100

    # loading the model
    if args.model_path[-3:] == '.py':
        model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path)
    else:
        assert False, "loading txt file model currently not supported"

    try: prior_samples = Predictive(model, num_samples = num_data)(jax.random.PRNGKey(0))
    except Exception as e: raise Exception("Error obtaining prior samples from model") from e

    _, syn_prior_encoded = postprocess_fn(prior_samples, df)

    train_data, num_train_data = preprocess_fn(syn_prior_encoded)

    assert isinstance(train_data, tuple)
    assert num_train_data == num_data # TODO: maybe not?

    posterior_params = train_model_no_dp(jax.random.PRNGKey(0),
        model, guide,
        train_data,
        batch_size = num_train_data//2,
        num_data = num_train_data,
        num_epochs = 3,
        silent = True
    )

    posterior_samples = Predictive(
        model, guide = guide, params = posterior_params,
        num_samples = num_train_data
    )(jax.random.PRNGKey(0))


if __name__ == "__main__":
    main()
