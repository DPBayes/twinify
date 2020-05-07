from collections import OrderedDict

import jax.numpy as np
import jax

import numpyro
import numpyro.distributions as dists
from numpyro.primitives import sample, param, deterministic
from numpyro.handlers import seed, trace
from dppp.minibatch import minibatch

from .mixture_model import MixtureModel

import numpy as onp

from typing import Type, Dict, Optional, Union


constraint_dtype_lookup = {
    dists.constraints._Boolean: 'bool',
    dists.constraints._Real: 'float',
    dists.constraints._GreaterThan: 'float',
    dists.constraints._Interval: 'float',
    dists.constraints._IntegerGreaterThan: 'int',
    dists.constraints._IntegerInterval: 'int',
    dists.constraints._Multinomial: 'int',
}
class Distribution:

    def __init__(self, name: str, numpyro_class: Type[dists.Distribution]) -> None:
        self._name = name
        self._numpyro_class = numpyro_class

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

    def instantiate(self, parameters: Optional[Dict[str, np.array]] = None) -> dists.Distribution:

        if parameters is None:
            parameters = self._make_zero_parameters()
        return self._numpyro_class(**parameters)

    def __call__(self, **parameters: Dict[str, np.array]) -> dists.Distribution:
        return self.instantiate(parameters)

    def _make_zero_parameters(self):
        zero_params = {
            param: transform(np.zeros(1))
            for param, transform in self.parameter_transforms.items()
        }
        return zero_params

    @staticmethod
    def get_support_dtype(dist: Union[Type[dists.Distribution], dists.Distribution]) -> str:
        """ Determines the type of a distribution's support

        Parameters:
            dist (dists.Distribution): numpyro Distribution instance
        Returns:
            (str) dtype of the given distribution's support
        """
        try:
            if not isinstance(dist.support, dists.constraints.Constraint):
                raise ValueError("Support of distribution of type {} cannot \
                    be derived from non-instantiated class".format(dist))
            support_constraint = type(dist.support)
        except AttributeError:
            support_constraint = type(dists.constraints.real)

        try:
            return constraint_dtype_lookup[support_constraint]
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
            return not isinstance(self.numpyro_class.support, dists.constraints.Constraint)
        except AttributeError:
            return True


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


##############################################################################################
#################### feature distribution automatization. WIP ################################

def extract_parameter_sites(feature_dists_and_shapes):
    """ Extract parameter sites and corresponding constraint transforms for each feature distribution

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
    Returns:
        OrderedDict(feature_name -> dict(parameter -> Transform))
    """
    params = {
        name: (dist.parameter_transforms, shape)
        for name, (dist, shape) in feature_dists_and_shapes.items()
    }
    return params

##################### automatic guide #######################################

def zip_dicts(first_dict, second_dict):
    """ Zips elements of two (ordered) dicts
    """
    return OrderedDict([
        (key, (first_dict[key], second_dict[key]))
        for key in first_dict
    ])

def make_guide(feature_params_and_shapes, k):
    """ Given parameter constraints and shapes, sets up and returns the guide function

    Parameters:
        feature_params_and_shapes (dict(feature_name -> (dict(parameter -> Transform), shape)))
    Returns:
        guide function
    """
    def guide(**kwargs):
        for name, (feature_parameters, shape) in feature_params_and_shapes.items():
            for parameter, transform in feature_parameters.items():
                loc = param("{}_{}_loc_uncons".format(name, parameter), .1*onp.random.randn(*shape))
                std = param("{}_{}_std_uncons".format(name, parameter), .1*onp.random.randn(*shape))
                dist = dists.TransformedDistribution(dists.Normal(loc, np.exp(std)), transform)
                sample("{}_{}".format(name, parameter), dist)

        pis_loc = param('pis_loc_uncons', .1*onp.random.randn(k - 1))
        pis_std = param('pis_std_uncons', .1*onp.random.randn(k - 1))

        pis_dist = dists.TransformedDistribution(dists.Normal(pis_loc, np.exp(pis_std)), dists.transforms.StickBreakingTransform())
        sample("pis", pis_dist)

    return guide

###################### automatic prior #############################


def parse_model(model_str, return_str_dict = False):
    """
    Parameters:
        model_str (str)
    Returns:
        OrderedDict(feature_name -> Distribution)
    """
    model = OrderedDict()
    model_str_dict = OrderedDict()
    for line in model_str.splitlines():
        # ignore comments (indicated by #)
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue

        parts = line.split(':')
        if len(parts) == 2:
            feature_name = parts[0].strip()
            distribution_name = parts[1].strip()

            dist = make_distribution(distribution_name)
            model[feature_name] = dist
            model_str_dict[feature_name] = dist.name

        else:
            # todo: something?
            pass
    if return_str_dict: return model, model_str_dict
    return model

##################### support lookup ########################

##################### prior lookup ########################

prior_lookup = {
    dists.Normal: {'loc': (dists.Normal, (0., 1.)), 'scale': (dists.Gamma, (2., 2.))},
    dists.BernoulliProbs: {'probs': (dists.Beta, (1., 1.))},
    dists.CategoricalProbs: {'probs': (dists.Dirichlet, (1.,))},
    dists.Poisson: {'rate': (dists.Exponential, (1.,))}
}

def create_model_prior_dists(feature_dists_and_shapes):
    """ Given feature distribution classes and shapes, create model prior distributions for parameters

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
    Returns:
        (dict(feature_name -> dict(parameter -> Distribution))
    """
    params = {
        name: {
            parameter: prior_dist(*[np.ones(shape) * prior_param for prior_param in prior_params])
            for parameter, (prior_dist, prior_params) in prior_lookup[feature_dist.numpyro_class].items()
        }
        for name, (feature_dist, shape) in feature_dists_and_shapes.items()
    }
    return params

def make_model(feature_dists_and_shapes, prior_dists, k):
    """ Given feature distribution classes and parameter priors, create a model function containing a mixture model

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
        prior_dists (dict(feature_name -> dict(parameter -> Distribution))
        k (int): number of mixture components
    """
    def model(N, num_obs_total=None):
        mixture_dists = []
        dtypes = []
        for feature_name, (feature_dist_class, _) in feature_dists_and_shapes.items():
            prior_values = {}
            feature_prior_dists = prior_dists[feature_name]
            for feature_prior_param, feature_prior_dist in feature_prior_dists.items():
                prior_values[feature_prior_param] = sample(
                    "{}_{}".format(feature_name, feature_prior_param),
                    feature_prior_dist
                )

            feature_dist = feature_dist_class(**prior_values)
            mixture_dists.append(feature_dist)
            dtypes.append(feature_dist_class.support_dtype)

        pis = sample('pis', dists.Dirichlet(np.ones(k)))
        with minibatch(N, num_obs_total=num_obs_total):
            mixture_model_dist = MixtureModel(mixture_dists, dtypes, pis)
            x = sample('x', mixture_model_dist, sample_shape=(N,))
            return x
    return model

def model_args_map(x, **kwargs):
    return (x.shape[0],), kwargs, {'x': x}

def guide_args_map(x, **kwargs):
    return (), kwargs, {}
