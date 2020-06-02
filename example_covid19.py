import jax.numpy as np
import jax

from dppp.modelling import sample_multi_posterior_predictive, make_observed_model
from numpyro.handlers import seed
from numpyro.contrib.autoguide import AutoDiagonalNormal


from twinify.infer import train_model, train_model_no_dp
import twinify.automodel as automodel

import numpy as onp

k = 7

model_str = """
crp: Normal
#ldh: Normal
covid_test: Bernoulli
#age: Poisson
severity: Categorical
"""

# crp: Normal, loc=.5

onp.random.seed(0)

features = automodel.parse_model(model_str)
feature_names = [feature.name for feature in features]

from covid19.data.preprocess_einstein import df
train_df = df[feature_names].dropna()

for feature in features:
    train_df[feature.name] = feature.preprocess_data(train_df[feature.name])

model = automodel.make_model(features, k)
seeded_model = seed(model, jax.random.PRNGKey(8365))

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
