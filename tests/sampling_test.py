import unittest
import pandas as pd
import jax
from twinify.sampling import sample_synthetic_data, reshape_and_postprocess_synthetic_data
from twinify.model_loading import guard_postprocess, DataDescription

import numpyro.distributions as dists
from numpyro import sample, plate

def model():
    mu = sample('mu', dists.Normal())
    with plate('batch', 10, 1):
        x = sample('x', dists.Normal(mu, 1).expand_by((1,5)).to_event(1))

def guide():
    sample('mu', dists.Delta(2.))

@guard_postprocess
def postprocess(samples, _: DataDescription):
    syn_data = samples['x']
    syn_df = pd.DataFrame(syn_data)
    return syn_df, syn_df

class SamplingTests(unittest.TestCase):

    def test_sampling(self) -> None:
        samples = sample_synthetic_data(model, guide, {}, jax.random.PRNGKey(0), 2, 3)
        self.assertEqual(set(samples.keys()), {'x'})
        self.assertEqual(samples['x'].shape, (2, 3, 5))

    def test_reshape_and_postprocess_combined(self) -> None:
        samples = sample_synthetic_data(model, guide, {}, jax.random.PRNGKey(0), 2, 3)

        prepared_postprocess = lambda samples: postprocess(samples, None)

        i = 0
        for syn_df, _ in reshape_and_postprocess_synthetic_data(
            samples, prepared_postprocess, separate_output=False, num_parameter_samples=2
        ):
            i += 1
            self.assertEqual(syn_df.values.shape, (2*3, 5))
        self.assertEqual(i, 1)

    def test_reshape_and_postprocess_separate(self) -> None:
        samples = sample_synthetic_data(model, guide, {}, jax.random.PRNGKey(0), 2, 3)

        prepared_postprocess = lambda samples: postprocess(samples, None)

        i = 0
        for syn_df, _ in reshape_and_postprocess_synthetic_data(
            samples, prepared_postprocess, separate_output=True, num_parameter_samples=2
        ):
            i += 1
            self.assertEqual(syn_df.values.shape, (3, 5))
        self.assertEqual(i, 2)
