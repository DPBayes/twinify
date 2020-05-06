import jax.numpy as np
import jax

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal


from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

k = 7
shapes = {
    'crp': (k,),
    'ldh': (k,),
    'covid_test': (k,),
    'age': (k,),
    'severity': (k, 4)
}

model_str = """
crp: Normal
#ldh: Normal
covid_test: Bernoulli
#age: Poisson
severity: Categorical
"""

# crp: Normal, loc=.5

onp.random.seed(0)

feature_dists = automodel.parse_model(model_str)
feature_dists_and_shapes = automodel.zip_dicts(feature_dists, shapes)

guide_param_sites = automodel.extract_parameter_sites(feature_dists_and_shapes)
guide = automodel.make_guide(guide_param_sites, k)
seeded_guide = seed(guide, jax.random.PRNGKey(23496))


prior_dists = automodel.create_model_prior_dists(feature_dists_and_shapes)
model = automodel.make_model(feature_dists_and_shapes, prior_dists, k)
seeded_model = seed(model, jax.random.PRNGKey(8365))

from covid19.data.preprocess_einstein import df
train_df = df[list(feature_dists.keys())].dropna()

guide = AutoDiagonalNormal(make_observed_model(model, automodel.model_args_map))

# import pandas as pd
# train_df = pd.DataFrame(onp.stack([onp.random.randn(10000), onp.random.binomial(1, .2, 10000)]).T, columns=['crp', 'covid_test'])

posterior_params = train_model(
    jax.random.PRNGKey(0),
    model, automodel.model_args_map, guide, None,
    train_df.to_numpy(),
    batch_size=100,
    num_epochs=1000,
    dp_scale=2.0
)

# posterior_samples = guide.sample_posterior(jax.random.PRNGKey(0), posterior_params, sample_shape=(1000,))

posterior_samples = sample_multi_posterior_predictive(jax.random.PRNGKey(0), 1000, model, (1,), guide, (), posterior_params)

pis = np.mean(posterior_samples['pis'], axis=0)
sev_probs = np.mean(posterior_samples['severity_probs'], axis=0)
sev_probs_marginal = np.sum(pis * sev_probs.T, axis=1)

syn_data = posterior_samples['x']
