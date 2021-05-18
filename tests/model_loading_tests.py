import unittest
import pandas as pd
import numpy as np
from numpyro.handlers import seed, trace
import jax
from twinify.model_loading import load_custom_numpyro_model, ModelException

class NumpyroModelLoadingTests(unittest.TestCase):

    def test_load_numpyro_model_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_custom_numpyro_model('./tests/models/does_not_exist')

    def test_load_numpyro_model_no_model_fn(self):
        with self.assertRaisesRegex(ModelException, "does not specify a 'model' function"):
            load_custom_numpyro_model('./tests/models/empty_model.py')

    def test_load_numpyro_model_not_a_module(self):
        with self.assertRaisesRegex(ModelException, "as a Python module"):
            load_custom_numpyro_model('./tests/models/gauss_data.csv')

    def test_load_numpyro_model_with_syntax_error(self):
        try:
            load_custom_numpyro_model('./tests/models/syntax_error.py')
        except ModelException as e:
            if isinstance(e.base, SyntaxError):
                return # = success here; otherwise, fall through to next line
        self.fail("load_custom_numpyro_model did not raise SyntaxError on model with syntax error")

    #### TESTS FOR POSTPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_postprocess(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        syn_data, encoded_syn_data = postprocess(samples, orig_data)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0], syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1], syn_data['second']))
        self.assertIsInstance(encoded_syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0] + 2, encoded_syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1] + 2, encoded_syn_data['second']))

    def test_load_numpyro_model_with_old_style_postprocess(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        syn_data, encoded_syn_data = postprocess(samples, orig_data)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0], syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1], syn_data['second']))
        self.assertIsInstance(encoded_syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0] + 2, encoded_syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1] + 2, encoded_syn_data['second']))

    def test_load_numpyro_model_with_broken_postprocess(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_broken.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if isinstance(e.base, KeyError) and e.title.find('postprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in postprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in postprocess")

    def test_load_numpyro_model_with_broken_old_style_postprocess(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style_broken.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if isinstance(e.base, KeyError) and e.title.find('postprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in postprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in postprocess")

    def test_load_numpyro_model_with_old_postprocess_but_assumed_new_model(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style.py')
        samples = {'first': np.zeros((10,)), 'second': np.zeros((10,))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('postprocessing function with a single argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong sample sites for old-style postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did no raise for wrong sample sites for old-style postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_signature(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_wrong_signature.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong signature in postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_returns(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_wrong_returns.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in postprocess")

    def test_load_numpyro_model_with_postprocess_old_style_wrong_returns(self):
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style_wrong_returns.py')
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            postprocess(samples, orig_data)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in postprocess")

    #### TESTS FOR PREPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_broken_preprocess(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_broken.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        try:
            preprocess(orig_data)
        except ModelException as e:
            if isinstance(e.base, KeyError) and e.title.find('preprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in preprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in preprocess")

    def test_load_numpyro_model_old_style_preprocess(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_old_style.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        train_data, num_data = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))

    def test_load_numpyro_model_preprocess_single_return(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_single_return.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        train_data, num_data = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))

    def test_load_numpyro_model_preprocess(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        train_data, num_data = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))

    def test_load_numpyro_model_with_preprocess_wrong_returns(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_wrong_returns.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in preprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in preprocess")

    def test_load_numpyro_model_with_preprocess_wrong_signature(self):
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_wrong_signature.py')
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
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
        model, guide, preprocess, postprocess = load_custom_numpyro_model('./tests/models/simple_gauss_model.py')
        self.assertIsNotNone(model)
        self.assertIsNotNone(guide)
        self.assertIsNotNone(preprocess)
        self.assertIsNotNone(postprocess)
        z = np.ones((10, 2))
        samples_with_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(z, num_obs_total=10)
        self.assertTrue(np.allclose(samples_with_obs['x']['value'], z))
        samples_no_obs = trace(seed(model, jax.random.PRNGKey(0))).get_trace(num_obs_total=10)
        self.assertEqual(samples_no_obs['x']['value'].shape, (2, ))
        self.assertFalse(np.allclose(samples_no_obs['x']['value'], z))

    def test_load_numpyro_model_broken_model(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_broken.py')
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z)
        except ModelException as e:
            if isinstance(e.base, NameError) and e.title.find('model'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_without_num_obs_total(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_no_num_obs_total.py')
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z, num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('num_obs_total') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_not_allowing_None_arguments(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_no_none.py')
        try:
            seed(model, jax.random.PRNGKey(0))(num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('None for synthesising data') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    # TODO: test handling of guides
    # TODO: some integrated tests to ensure preprocess-model-postprocess pipeline error are handled well?