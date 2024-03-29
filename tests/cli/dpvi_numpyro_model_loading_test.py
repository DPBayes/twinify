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

import unittest
import pandas as pd
import numpy as np
from numpyro.handlers import seed, trace
import jax
from twinify.cli.dpvi_numpyro_model_loading import load_custom_numpyro_model, ModelException
from twinify import DataDescription
from argparse import Namespace

class NumpyroModelLoadingTests(unittest.TestCase):

    def test_load_numpyro_model_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_custom_numpyro_model('./tests/cli/models/does_not_exist', Namespace(), [], DataDescription(dict()))

    def test_load_numpyro_model_no_model_fn(self):
        with self.assertRaisesRegex(ModelException, "does neither specify a 'model'"):
            load_custom_numpyro_model('./tests/cli/models/empty_model.py', Namespace(), [], DataDescription(dict()))

    def test_load_numpyro_model_not_a_module(self):
        with self.assertRaisesRegex(ModelException, "as a Python module"):
            load_custom_numpyro_model('./tests/cli/models/gauss_data.csv', Namespace(), [], DataDescription(dict()))

    def test_load_numpyro_model_with_syntax_error(self):
        try:
            load_custom_numpyro_model('./tests/cli/models/syntax_error.py', Namespace(), [], DataDescription(dict()))
        except ModelException as e:
            if isinstance(e.base, SyntaxError):
                return # = success here; otherwise, fall through to next line
        self.fail("load_custom_numpyro_model did not raise SyntaxError on model with syntax error")

    #### TESTS FOR POSTPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_postprocess(self):
        samples = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(samples)
        _, _, _, postprocess = load_custom_numpyro_model('./tests/cli/models/postprocess.py', Namespace(), [], data_description)
        syn_data = postprocess(samples)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['first'] + 2, syn_data['first']))
        self.assertTrue(np.allclose(samples['second'] + 2, syn_data['second']))
        self.assertTrue(np.allclose(samples['first'], syn_data['new_first']))

    def test_load_numpyro_model_with_broken_postprocess(self):
        samples = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(samples)
        _, _, _, postprocess = load_custom_numpyro_model('./tests/cli/models/postprocess_broken.py', Namespace(), [], data_description)
        try:
            postprocess(samples)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if isinstance(e.base, KeyError) and e.title.find('postprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in postprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_signature(self):
        samples = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(samples)
        _, _, _, postprocess = load_custom_numpyro_model('./tests/cli/models/postprocess_wrong_signature.py', Namespace(), [], data_description)
        try:
            postprocess(samples)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong signature in postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_returns(self):
        samples = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(samples)
        _, _, _, postprocess = load_custom_numpyro_model('./tests/cli/models/postprocess_wrong_returns.py', Namespace(), [], data_description)
        try:
            postprocess(samples)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in postprocess")

    #### TESTS FOR PREPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_broken_preprocess(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/cli/models/preprocess_broken.py', Namespace(), [], data_description)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if isinstance(e.base, KeyError) and e.title.find('preprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in preprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in preprocess")

    def test_load_numpyro_model_preprocess_return_series(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/cli/models/preprocess_return_series.py', Namespace(), [], data_description)
        train_data = preprocess(orig_data)
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data['new_first']))
        self.assertEqual(10, len(train_data))

    def test_load_numpyro_model_with_preprocess_wrong_returns(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/cli/models/preprocess_wrong_returns.py', Namespace(), [], data_description)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in preprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in preprocess")

    def test_load_numpyro_model_with_preprocess_wrong_signature(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/cli/models/preprocess_wrong_signature.py', Namespace(), [], data_description)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in preprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong signature in preprocess")

    #### TESTING MODEL LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_simple_working_model(self):
        """ only verifies that no errors occur and all returned functions are not None """
        model, guide, preprocess, postprocess = load_custom_numpyro_model('./tests/cli/models/simple_gauss_model.py', Namespace(), [], DataDescription(dict()))
        self.assertIsNotNone(model)
        self.assertIsNotNone(guide)
        self.assertIsNotNone(preprocess)
        self.assertIsNotNone(postprocess)
        z = np.ones((10, 2))
        samples_with_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertTrue(np.allclose(samples_with_obs['x']['value'], z))
        samples_no_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(samples_no_obs['x']['value'].shape, (1, 2))
        self.assertFalse(np.allclose(samples_no_obs['x']['value'], z))

    def test_load_numpyro_model_broken_model(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/cli/models/simple_gauss_model_broken.py', Namespace(), [], DataDescription(dict()))
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z)
        except ModelException as e:
            if isinstance(e.base, NameError) and e.title.find('model'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_without_num_obs_total(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/cli/models/simple_gauss_model_no_num_obs_total.py', Namespace(), [], DataDescription(dict()))
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z, num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('num_obs_total') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_not_allowing_None_arguments(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/cli/models/simple_gauss_model_no_none.py', Namespace(), [], DataDescription(dict()))
        try:
            seed(model, jax.random.PRNGKey(0))(num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('None for synthesising data') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_not_a_function(self):
        try:
            load_custom_numpyro_model('./tests/cli/models/model_not_a_function.py', Namespace(), [], DataDescription(dict()))
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('must be a function') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for model not being a function, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for model not being a function")


    #### TESTING MODEL FACTORY
    def test_load_numpyro_model_model_factory(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/cli/models/model_factory.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(guide)
        self.assertIsNotNone(preprocess)
        self.assertIsNotNone(postprocess)
        z = orig_data.to_numpy()
        samples_with_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertTrue(np.allclose(samples_with_obs['x']['value'], z))
        samples_no_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(samples_no_obs['x']['value'].shape, (1, 2))
        self.assertFalse(np.allclose(samples_no_obs['x']['value'], z))

    def test_load_numpyro_model_model_factory_with_guide(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/cli/models/model_factory_with_guide.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(guide)
        self.assertIsNotNone(preprocess)
        self.assertIsNotNone(postprocess)
        z = orig_data.to_numpy()
        guide_samples_with_obs = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertEqual(guide_samples_with_obs['mu']['value'].shape, (2,))
        self.assertEqual(guide_samples_with_obs['sigma']['value'].shape, (2,))
        guide_samples_no_obs = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(guide_samples_no_obs['mu']['value'].shape, (2,))
        self.assertEqual(guide_samples_no_obs['sigma']['value'].shape, (2,))

        samples_with_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertTrue(np.allclose(samples_with_obs['x']['value'], z))
        samples_no_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(samples_no_obs['x']['value'].shape, (1, 2))
        self.assertFalse(np.allclose(samples_no_obs['x']['value'], z))

    def test_load_numpyro_model_model_factory_with_autoguide(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/cli/models/model_factory_with_autoguide.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(guide)
        self.assertIsNotNone(preprocess)
        self.assertIsNotNone(postprocess)
        z = orig_data.to_numpy()
        guide_samples_with_obs = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertEqual(guide_samples_with_obs['guide_loc']['value'].shape, (4,)) # 2 parameters (mu, sigma) with 2 dimensions each
        self.assertEqual(guide_samples_with_obs['guide_scale']['value'].shape, (4,))
        guide_samples_no_obs = trace(seed(guide, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(guide_samples_no_obs['guide_loc']['value'].shape, (4,))
        self.assertEqual(guide_samples_no_obs['guide_scale']['value'].shape, (4,))

        samples_with_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertTrue(np.allclose(samples_with_obs['x']['value'], z))
        samples_no_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(samples_no_obs['x']['value'].shape, (1, 2))
        self.assertFalse(np.allclose(samples_no_obs['x']['value'], z))

    def test_load_numpyro_model_model_factory_broken(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        try:
            load_custom_numpyro_model(
                './tests/cli/models/model_factory_broken.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
            )
        except ModelException as e:
            print(e.title)
            if e.title.find('model factory'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model_factory, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model_factory")

    def test_load_numpyro_model_model_factory_wrong_signature(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        try:
            load_custom_numpyro_model(
                './tests/cli/models/model_factory_wrong_signature.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong signature in model_factory")

    def test_load_numpyro_model_model_factory_wrong_returns_none(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        try:
            load_custom_numpyro_model(
                './tests/cli/models/model_factory_wrong_returns_none.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('either a model function or a tuple') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong returns in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong returns in model_factory")

    def test_load_numpyro_model_model_factory_wrong_returns_bad_tuple(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        data_description = DataDescription.from_dataframe(orig_data)
        try:
            load_custom_numpyro_model(
                './tests/cli/models/model_factory_wrong_returns_bad_tuple.py', Namespace(epsilon=1.), ['--prior_mu', '10'], data_description
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('either a model function or a tuple') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong returns in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong returns in model_factory")


    # TODO: test handling of guides
    # TODO: some integrated tests to ensure preprocess-model-postprocess pipeline error are handled well?