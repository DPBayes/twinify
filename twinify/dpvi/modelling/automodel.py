# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022- twinify Developers and their Assignees

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
Automatic modelling routines used by twinify main script.
"""

from collections import OrderedDict

import jax.numpy as jnp
import jax
from jax.dtypes import canonicalize_dtype, issubdtype

import numpyro
import numpyro.distributions as dists
from numpyro.primitives import sample, param, deterministic, plate
from numpyro.handlers import seed, trace
from d3p.util import unvectorize_shape_2d

from .mixture_model import MixtureModel
from .na_model import NAModel

import numpy as np

from typing import Type, Dict, Optional, Union, Tuple, List, Callable, Sequence
from abc import ABCMeta, abstractmethod

import pandas as pd


constraint_dtype_lookup = {
    dists.constraints._Boolean: 'int',
    dists.constraints._Real: 'float',
    dists.constraints._GreaterThan: 'float',
    dists.constraints._Interval: 'float',
    dists.constraints._IntegerGreaterThan: 'int',
    dists.constraints._IntegerInterval: 'int',
    dists.constraints._Multinomial: 'int',
}

class Distribution:
    """ Wraps a numpyro distribution and exposes relevant properties for
        automated model building
    """

    def __init__(self, name: str, numpyro_class: Type[dists.Distribution]) -> None:
        self._name = name
        self._numpyro_class = numpyro_class
        self._is_categorical = numpyro_class in categorical_dist_lookup
        assert (not self.is_categorical or self.is_discrete), \
                "A continuos distribution cannot be categorical"

    @staticmethod
    def from_distribution_instance(dist: dists.Distribution) -> 'Distribution':
        dist_class = type(dist)
        assert(issubclass(dist_class, dists.Distribution))

        try:
            name = dist_class.__name__
        except:
            name = ''

        return Distribution(name, dist_class)

    @property
    def name(self) -> str:
        return self._name

    @property
    def numpyro_class(self) -> Type[dists.Distribution]:
        return self._numpyro_class

    @property
    def parameter_transforms(self) -> Dict[str, dists.transforms.Transform]:
        param_transforms = {
            param: dists.transforms.biject_to(constraint)
            for param, constraint in self._numpyro_class.__dict__['arg_constraints'].items()
        }
        return param_transforms

    def instantiate(self, parameters: Optional[Dict[str, jnp.array]] = None) -> dists.Distribution:

        if parameters is None:
            parameters = self._make_zero_parameters()
        return self._numpyro_class(**parameters)

    def __call__(self, **parameters: Dict[str, jnp.array]) -> dists.Distribution:
        return self.instantiate(parameters)

    def _make_zero_parameters(self):
        zero_params = {
            param: transform(jnp.zeros(1))
            for param, transform in self.parameter_transforms.items()
        }
        return zero_params

    @staticmethod
    def get_support_constraint(dist: Union[Type[dists.Distribution], dists.Distribution]) -> str:
        support_constraint = type(dists.constraints.real)
        if dist.support is not None:
            if not isinstance(dist.support, dists.constraints.Constraint):
                raise ValueError("Support of distribution of type {} cannot \
                    be derived from non-instantiated class".format(dist))
            support_constraint = type(dist.support)
        return support_constraint


    @staticmethod
    def get_support_dtype(dist: Union[Type[dists.Distribution], dists.Distribution]) -> str:
        """ Determines the type of a distribution's support

        Parameters:
            dist (dists.Distribution): numpyro Distribution instance
        Returns:
            (str) dtype of the given distribution's support
        """
        support_constraint = Distribution.get_support_constraint(dist)

        try:
            return canonicalize_dtype(constraint_dtype_lookup[support_constraint])
        except KeyError:
            return ValueError("A distribution with support {} is currently \
                not supported".format(support_constraint))

    @property
    def support_dtype(self) -> str:
        if self.has_data_dependent_shape:
            return self.get_support_dtype(self.instantiate())
        else:
            return self.get_support_dtype(self.numpyro_class)

    @property
    def has_data_dependent_shape(self) -> bool:
        try:
            return dists.constraints.is_dependent(self.numpyro_class.support)
        except AttributeError:
            return True

    @property
    def is_categorical(self) -> bool:
        return self._is_categorical

    @property
    def is_discrete(self) -> bool:
        return self.get_support_constraint(self.numpyro_class).is_discrete


class ModelFeature:
    """ Represents how a feature in the data is modelled
    """

    def __init__(self, feature_name: str, dist: Distribution) -> None:
        self._name = feature_name
        self._dist = dist
        self._shape = None
        self._value_map = None
        self._missing_values = False

    @staticmethod
    def from_distribution_instance(feature_name: str, dist: dists.Distribution) -> 'ModelFeature':
        meta_dist = Distribution.from_distribution_instance(dist)
        shape = dist.batch_shape
        feature = ModelFeature(feature_name, meta_dist)
        feature.shape = shape
        return feature

    @property
    def name(self) -> str:
        """ Feature name """
        return self._name

    @property
    def distribution(self) -> Distribution:
        """ automodel.Distribution the feature is modelled by """
        return self._dist

    @property
    def shape(self) -> Tuple[int]:
        """ Simple shape of parameters for this feature distribution.

        Does not include mixture dimension.
        """
        if not self.distribution.has_data_dependent_shape:
            return tuple()
        elif self._shape is None:
            raise RuntimeError("The shape of the distribution depends on the data\
                but not data has been preprocessed yet.")
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int]):
        self._shape = value

    @property
    def has_missing_values(self) -> bool:
        return self._missing_values

    def preprocess_data(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """ Preprocesses data according to the feature distribution.

        If a pd.Series instance is passed in, it will be treated as the feature
        data. If a pd.DataFrame is passed in, this function will only affect
        the column corresponding the feature's name (self.name).

        If the feature follows a Categorical distribution, this will extract
        the number of categories present in the data.

        If the data type is not numeric for any distribution modelling some
        kind of categories (e.g., Categorical, Bernoulli), it will be mapped
        to integer categories.
        """
        column = data
        if isinstance(data, pd.DataFrame):
            column = data[self.name]
        assert(isinstance(column, pd.Series))

        # detect whether there are missing values in the data
        if np.any(column.isna()):
            self._missing_values = True

        # TODO: with the introduction of DataDescription, this is probably somewhat redundant now - need to test
        # if the distribution is a categorical, we need to determine the number
        # of categories
        if self.distribution.is_categorical:

            # if shape of the categorical distribution depends on data, we extract the
            #    number of unique feature values in the data
            if self.distribution.has_data_dependent_shape:
                self.shape = (len(column.dropna().unique()), )

            # we map the feature values to integers
            self._value_map = {val: i for i, val in enumerate(column.dropna().unique())}
            column = column.map(self._value_map)

        if isinstance(data, pd.DataFrame):
            data[self.name] = column
        else:
            data = column
        return data

    def postprocess_data(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        column = data
        if isinstance(data, pd.DataFrame):
            column = data[self.name]
        assert(isinstance(column, pd.Series))

        if self._value_map is not None:
            inverse_map = {v: k for k, v in self._value_map.items()}
            column = column.map(inverse_map)

        if isinstance(data, pd.DataFrame):
            data[self.name] = column
        else:
            data = column
        return data

    def instantiate(self, **params):
        return self.distribution(**params)

    def __repr__(self) -> str:
        return f"ModelFeature({self.name}, {self.distribution}, {self.shape})"

################################# model parsing ################################


categorical_dist_lookup = {
    dists.BernoulliProbs, dists.BernoulliLogits,
    dists.CategoricalProbs, dists.CategoricalLogits,
    dists.MultinomialProbs, dists.MultinomialLogits,
    dists.OrderedLogistic
}

dist_lookup = {
    "normal": Distribution("Normal", dists.Normal),
    "bernoulli": Distribution("Bernoulli", dists.BernoulliProbs),
    "categorical": Distribution("Categorical", dists.CategoricalProbs),
    "poisson": Distribution("Poisson", dists.Poisson)
}


def make_distribution(token: str) -> Distribution:
    try:
        return dist_lookup[token.lower()]
    except KeyError:
        raise ValueError("{} distributions are currently not supported".format(token))

class ParsingError(Exception):
    pass

def parse_model(model_str: str) -> List[ModelFeature]:
    """
    Parameters:
        model_str (str)
    Returns:
        OrderedDict(feature_name -> Distribution)
    """
    features = []
    for line_nr, line in enumerate(model_str.splitlines()):
        # ignore comments (indicated by #)
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue

        parts = line.split(':')
        if len(parts) == 2:
            feature_name = parts[0].strip()
            distribution_name = parts[1].strip()

            try:
                dist = make_distribution(distribution_name)
                features.append(ModelFeature(feature_name, dist))
            except ValueError as e:
                raise ParsingError("Unable to parse line {}:\n<< {} >>\n{}".format(
                    line_nr, line, e
                ))

        else:
            raise ParsingError("Unable to parse line {}:\n<< {} >>\nSyntax error!".format(
                line_nr, line
            ))
    return features

def extract_features_from_mixture_model(mixture: MixtureModel, feature_names: List[str]) -> List[ModelFeature]:
    features = [ModelFeature.from_distribution_instance(name, dist)
                for name, dist in zip(feature_names, mixture.distributions)]
    return features


def parse_model_fn(model: Callable, feature_names: List[str]) -> List[ModelFeature]:
    model_trace = trace(seed(model, rng_seed=0)).get_trace(1)
    assert('x' in model_trace)
    mixture_model = model_trace['x']['fn']
    return extract_features_from_mixture_model(mixture_model, feature_names)


##################### prior lookup ########################

prior_lookup = {
    dists.Normal: {'loc': (dists.Normal, (0., 10.)), 'scale': (dists.LogNormal, (0., 2.))},
    dists.BernoulliProbs: {'probs': (dists.Beta, (1., 1.))},
    dists.CategoricalProbs: {'probs': (dists.Dirichlet, (1.,))},
    dists.Poisson: {'rate': (dists.Exponential, (1.,))}
}

def create_feature_prior_dists(feature: ModelFeature, k: int) -> Dict[str, dists.Distribution]:
    shape = (k,) + feature.shape
    prior_dists = {
        parameter: prior_dist(*[jnp.ones(shape) * prior_param for prior_param in prior_params])
        for parameter, (prior_dist, prior_params) in prior_lookup[feature.distribution.numpyro_class].items()
    }
    return prior_dists

################### typed dist ##########################
class TypedDistribution(dists.Distribution):

    def __init__(self, base_dist, dtype, validate_args=None):
        super(TypedDistribution, self).__init__(base_dist.batch_shape, base_dist.event_shape, validate_args=validate_args)
        self.dtype = dtype
        self.base_dist = base_dist

    @property
    def support(self):
        return self.base_dist.support

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value.astype(self.dtype))
        return log_prob

    def sample(self, key, sample_shape=()):
        return self.base_dist.sample(key, sample_shape=sample_shape)

    def __repr__(self) -> str:
        return f"TypedDistribution({self.base_dist})"


################### automatically build mixture model ##########################

def make_model(features: List[ModelFeature], k: int) -> Callable[..., None]:
    """ Given model feature specifications, create a model function for a mixture model

    Parameters:
        features: list of ModelFeature instances, specifying the feature distributions
        k: number of mixture components
    """
    def model(x=None, num_obs_total=None):
        assert x is None or len(jnp.shape(x)) == 2
        if x is None:
            N = 1
        else:
            N = jnp.shape(x)[0]
        if num_obs_total is None:
            num_obs_total = N

        assert isinstance(num_obs_total, int) and num_obs_total > 0
        assert N <= num_obs_total

        mixture_dists = []
        for feature in features:
            prior_values = {}
            feature_prior_dists = create_feature_prior_dists(feature, k)
            for feature_prior_param, feature_prior_dist in feature_prior_dists.items():
                prior_values[feature_prior_param] = sample(
                    "{}_{}".format(feature.name, feature_prior_param),
                    feature_prior_dist
                )

            feature_dist = feature.instantiate(**prior_values)
            feature_dist = TypedDistribution(feature_dist, feature.distribution.support_dtype)
            if feature.has_missing_values:
                feature_na_prob = sample(
                    "{}_na_prob".format(feature.name),
                    dists.Beta(2.*jnp.ones(k), 2.*jnp.ones(k))
                )
                feature_dist = NAModel(feature_dist, feature_na_prob)

            mixture_dists.append(feature_dist)

        pis = sample('pis', dists.Dirichlet(jnp.ones(k)))
        with plate('batch', num_obs_total, N):
            mixture_model_dist = MixtureModel(mixture_dists, pis)
            x = sample('x', mixture_model_dist, obs=x)
            return x
    return model

####################### postprocessing ###########################
def postprocess_function_factory(features):
    def postprocess_fn(syn_df: pd.DataFrame) -> pd.DataFrame:
        encoded_syn_df = syn_df.copy()
        for feature in features:
            encoded_syn_df = feature.postprocess_data(encoded_syn_df)
        return encoded_syn_df
    return postprocess_fn
