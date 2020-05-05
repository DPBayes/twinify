from collections import OrderedDict

import jax.numpy as np
import jax

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import sample, param, deterministic
from numpyro.handlers import seed, trace
from dppp.minibatch import minibatch

from .mixture_model import MixtureModel

import numpy as onp

##############################################################################################
#################### feature distribution automatization. WIP ################################


# from functools import reduce
# def create_parameter_sites(feature_dists, k = 1):
#     """Initializes parameter values given a dictionary of (feature name, distribution) pairs.

#     Determines all parameter sites for each feature distribution and initializes them to neutral
#     values (zero transformed to the parameter domain).

#     Currently only supports independently modelled features.

#     Parameters:
#         feature_dists (dict(feature_name -> Distribution)): dictionary associating features names with
#             numpyro Distribution objects
#         k (int): number of mixture components
#     Returns:
#         dict(feature_name -> dict(parameter -> jax.numpy.array)): dictionary associationg feature names
#             with initialized parameters for a k-fold mixture model
#     """
#     # todo(lumip): problem: for Categorical Probs, need to specify how many categories there are
#     #    (currenty uses mixture component number k, which is very much not correct)..
#     def create_parameter_sites_for_feature(dst: dist.Distribution, k: int = 1):
#         untransformed_zero = np.zeros((k,))
#         params = {
#             param: dist.transforms.biject_to(constraint) (untransformed_zero)
#             for param, constraint in dst.__dict__['arg_constraints'].items()
#         }
#         return params

#     params = {
#         name: create_parameter_sites_for_feature(dist, k)
#         for name, dist in feature_dists.items()
#     }
#     return params

def extract_parameter_sites(feature_dists_and_shapes):
    """ Extract parameter sites and corresponding constraint transforms for each feature distribution

    Parameters:
        feature_dists_and_shapes (OrderedDict(feature_name -> (Distribution, shape)))
    Returns:
        OrderedDict(feature_name -> dict(parameter -> Transform))
    """
    def create_parameter_sites_for_feature(dst: dist.Distribution):
        params = {
            param: dist.transforms.biject_to(constraint)
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

def create_guide_dists(feature_params_and_shapes):
    """ Given parameter constraints and shapes, creates distribution objects for parameter guides
    Parameters:
        feature_params_and_shapes (dict(feature_name -> (dict(parameter -> Transform), shape)))
    Returns:
        dict(feature_name -> dict(parameter -> Distribution))
    """
    params = {
        name: {
            parameter: dist.TransformedDistribution(dist.Normal(
                # todo(lumip): deal with transforms changing shape #shapeshifters
                param("{}_{}_loc_uncons".format(name, parameter), onp.random.randn(*shape)),
                param("{}_{}_std_uncons".format(name, parameter), np.exp(onp.random.randn(*shape))) ## terrible code!!!
            ), transform)
            for parameter, transform in feature_parameters.items()
        }
        for name, (feature_parameters, shape) in feature_params_and_shapes.items()
    }
    return params

def make_guide(guide_dists):
    """ Given distribution instances for parameter guides, sets up and returns the guide function

    Parameters:
        guide_dists (dict(feature_name -> dict(parameter -> Distribution))
    Returns:
        guide function
    """
    def guide(**kwargs):
        for name, feature_parameters in guide_dists.items():
            for parameter, dist in feature_parameters.items():
                sample("{}_{}".format(name, parameter), dist)
    return guide

###################### automatic prior #############################

dist_lookup = {
    "Normal": dist.Normal,
    "Bernoulli": dist.BernoulliProbs,
    "Categorical": dist.CategoricalProbs,
    "Poisson": dist.Poisson
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


prior_lookup = {
    dist.Normal: {'loc': (dist.Normal, (0., 1.)), 'scale': (dist.Gamma, (2., 2.))},
    dist.BernoulliProbs: {'probs': (dist.Beta, (1., 1.))},
    dist.CategoricalProbs: {'probs': (dist.Dirichlet, (1.,))},
    dist.Poisson: {'rate': (dist.Exponential, (1.,))}
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

        pis = sample('pis', dist.Dirichlet(np.ones(k)))
        with minibatch(N, num_obs_total=num_obs_total):
            mixture_model_dist = MixtureModel(mixture_dists, dtypes, pis)
            x = sample('x', mixture_model_dist, sample_shape=(N,))
            return x
    return model

def model_args_map(x, **kwargs):
    return (x.shape[0],), kwargs, {'x': x}

def guide_args_map(x, **kwargs):
    return (), kwargs, {}
