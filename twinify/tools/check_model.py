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

from jax.config import config
config.update("jax_enable_x64", True)
import jax

import pandas as pd
from numpyro.infer import Predictive
from twinify.infer import train_model_no_dp
from twinify.model_loading import ModelException, NumpyroModelParsingException, load_custom_numpyro_model

def setup_argument_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('data_path', type=str, help='Path to input data.')
    parser.add_argument('model_path', type=str, help='Path to model file (.txt or .py).')
    parser.add_argument("--drop_na", default=False, action='store_true', help="Remove missing values from data.")


def main(args: argparse.Namespace):
    # read data
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print("#### UNABLE TO READ DATA FILE ####")
        print(e)
        exit(1)

    train_df = df.copy()
    if args.drop_na:
        train_df = train_df.dropna()
    num_data = 100

    try:

        # loading the model
        if args.model_path[-3:] == '.py':
            try:
                model, guide, preprocess_fn, postprocess_fn = load_custom_numpyro_model(args.model_path)
            except (ModuleNotFoundError, FileNotFoundError) as e:
                print("#### COULD NOT FIND THE MODEL FILE ####")
                print(e)
                exit(1)
            except NumpyroModelParsingException as e:
                raise e
        else:
            print("#### loading txt file model currently not supported ####")
            exit(2)

        try: prior_samples = Predictive(model, num_samples = num_data)(jax.random.PRNGKey(0))
        except Exception as e: raise ModelException("Error while obtaining prior samples from model", base_exception=e)

        _, syn_prior_encoded = postprocess_fn(prior_samples, df)

        train_data, num_train_data = preprocess_fn(syn_prior_encoded)

        assert isinstance(train_data, tuple)
        assert num_train_data == num_data # TODO: maybe not?

        try:
            posterior_params = train_model_no_dp(jax.random.PRNGKey(0),
                model, guide,
                train_data,
                batch_size = num_train_data//2,
                num_data = num_train_data,
                num_epochs = 3,
                silent = True
            )
        except Exception as e:
            raise ModelException("Error while performing inference", base_exception=e)

        try:
            posterior_samples = Predictive(
                model, guide = guide, params = posterior_params,
                num_samples = num_train_data
            )(jax.random.PRNGKey(0))
        except Exception as e:
            raise ModelException("Error while obtaining posterior samples from model", base_exception=e)

        print("Everything okay!")
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
    parser = argparse.ArgumentParser(description='Twinify: Program for creating synthetic twins under differential privacy.',\
        fromfile_prefix_chars="%")
    setup_argument_parser(parser)
    exit(
        main(parser.parse_args())
    )
