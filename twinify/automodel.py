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

##############################################################################################
#################### feature distribution automatization. WIP ################################

def extract_parameter_sites(feature_dists_and_shapes):
    """ Extract parameter sites and corresponding constraint transforms for each feature distribution

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
    Returns:
        OrderedDict(feature_name -> dict(parameter -> Transform))
    """
    def create_parameter_sites_for_feature(dst: dists.Distribution):
        params = {
            param: dists.transforms.biject_to(constraint)
            for param, constraint in dst.__dict__['arg_constraints'].items()
        }
        return params

    params = {
        name: (create_parameter_sites_for_feature(dist), shape)
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

dist_lookup = {
    "Normal": dists.Normal,
    "Bernoulli": dists.BernoulliProbs,
    "Categorical": dists.CategoricalProbs,
    "Poisson": dists.Poisson
}

}

def parse_model(model_str):
    """
    Parameters:
        model_str (str)
    Returns:
        OrderedDict(feature_name -> Distribution)
    """
    model = OrderedDict()
    for line in model_str.splitlines():
        # ignore comments (indicated by #)
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue

        parts = line.split(':')
        if len(parts) == 2:
            feature_name = parts[0].strip()
            distribution = parts[1].strip()

            if distribution in dist_lookup.keys():
                model[feature_name] = dist_lookup[distribution]
        else:
            # todo: something?
            pass
    return model

##################### support lookup ########################
support_lookup = {
    "Normal": "float",
    "Bernoulli": "bool",
    "Categorical": "int",
    "Poisson": "int"
}

def parse_support(model_str):
    """
    Parameters:
        model_str (str)
    Returns:
        OrderedDict(feature_name -> Distribution)
    """
    support = []
    for line in model_str.splitlines():
        # ignore comments (indicated by #)
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue

        parts = line.split(':')
        if len(parts) == 2:
            feature_name = parts[0].strip()
            distribution = parts[1].strip()

            if distribution in support_lookup.keys():
                support.append(support_lookup[distribution])
        else:
            # todo: something?
            pass
    return support

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
            # parameter: sample(
            #     "{}_{}".format(name, parameter),
            #     prior_dist(*[np.ones(*shape) * prior_param for prior_param in prior_params])
            # )
            for parameter, (prior_dist, prior_params) in prior_lookup[feature_dist].items()
        }
        for name, (feature_dist, shape) in feature_dists_and_shapes.items()
    }
    return params

def make_model(feature_dists_and_shapes, prior_dists, dtypes, k):
    """ Given feature distribution classes and parameter priors, create a model function containing a mixture model

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
        prior_dists (dict(feature_name -> dict(parameter -> Distribution))
        dtypes (list(str)): dtype of data for each feature
        k (int): number of mixture components
    """
    def model(N, num_obs_total=None):
        mixture_dists = []
        for feature_name, (feature_dist, _) in feature_dists_and_shapes.items():
            prior_values = {}
            feature_prior_dists = prior_dists[feature_name]
            for feature_prior_param, feature_prior_dist in feature_prior_dists.items():
                prior_values[feature_prior_param] = sample(
                    "{}_{}".format(feature_name, feature_prior_param),
                    feature_prior_dist
                )

            mixture_dists.append(feature_dist(**prior_values))

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
