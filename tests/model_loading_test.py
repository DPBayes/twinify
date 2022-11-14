import unittest
import pandas as pd
import numpy as np
from numpyro.handlers import seed, trace
import jax
from twinify.cli.model_loading import load_custom_numpyro_model, ModelException
from argparse import Namespace

class NumpyroModelLoadingTests(unittest.TestCase):

    def test_load_numpyro_model_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_custom_numpyro_model('./tests/models/does_not_exist', Namespace(), [], pd.DataFrame())

    def test_load_numpyro_model_no_model_fn(self):
        with self.assertRaisesRegex(ModelException, "does neither specify a 'model'"):
            load_custom_numpyro_model('./tests/models/empty_model.py', Namespace(), [], pd.DataFrame())

    def test_load_numpyro_model_not_a_module(self):
        with self.assertRaisesRegex(ModelException, "as a Python module"):
            load_custom_numpyro_model('./tests/models/gauss_data.csv', Namespace(), [], pd.DataFrame())

    def test_load_numpyro_model_with_syntax_error(self):
        try:
            load_custom_numpyro_model('./tests/models/syntax_error.py', Namespace(), [], pd.DataFrame())
        except ModelException as e:
            if isinstance(e.base, SyntaxError):
                return # = success here; otherwise, fall through to next line
        self.fail("load_custom_numpyro_model did not raise SyntaxError on model with syntax error")

    #### TESTS FOR POSTPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_postprocess(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess.py', Namespace(), [], orig_data)
        syn_data, encoded_syn_data = postprocess(samples, orig_data, feature_names)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0], syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1], syn_data['second']))
        self.assertIsInstance(encoded_syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0] + 2, encoded_syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1] + 2, encoded_syn_data['second']))

    def test_load_numpyro_model_with_postprocess_multiple_sample_sites(self):
        samples = {'x': np.zeros((10,2)), 'y': np.zeros((10,))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_multiple_sample_sites.py', Namespace(), [], orig_data)
        syn_data, encoded_syn_data = postprocess(samples, orig_data, feature_names)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0], syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1], syn_data['second']))
        self.assertTrue(np.allclose(samples['y'], syn_data['foo']))
        self.assertIsInstance(encoded_syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0] + 2, encoded_syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1] + 2, encoded_syn_data['second']))
        self.assertTrue(np.allclose(samples['y'] + 2, encoded_syn_data['foo']))

    def test_load_numpyro_model_with_old_style_postprocess(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style.py', Namespace(), [], orig_data)
        syn_data, encoded_syn_data = postprocess(samples, orig_data, feature_names)
        self.assertIsInstance(syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0], syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1], syn_data['second']))
        self.assertIsInstance(encoded_syn_data, pd.DataFrame)
        self.assertTrue(np.allclose(samples['x'][:,0] + 2, encoded_syn_data['first']))
        self.assertTrue(np.allclose(samples['x'][:,1] + 2, encoded_syn_data['second']))

    def test_load_numpyro_model_with_broken_postprocess(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_broken.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if isinstance(e.base, KeyError) and e.title.find('postprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in postprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in postprocess")

    def test_load_numpyro_model_with_broken_old_style_postprocess(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style_broken.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if isinstance(e.base, KeyError) and e.title.find('postprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in postprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in postprocess")

    def test_load_numpyro_model_with_old_postprocess_but_assumed_new_model(self):
        samples = {'first': np.zeros((10,)), 'second': np.zeros((10,))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('postprocessing function with a single argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong sample sites for old-style postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did no raise for wrong sample sites for old-style postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_signature(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_wrong_signature.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e: # check exception is raised
            # and original exception is passed on correctly
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong signature in postprocess")

    def test_load_numpyro_model_with_postprocess_wrong_returns(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_wrong_returns.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in postprocess")

    def test_load_numpyro_model_with_postprocess_old_style_wrong_returns(self):
        samples = {'x': np.zeros((10,2))}
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        feature_names = ['first', 'second']
        _, _, _, postprocess = load_custom_numpyro_model('./tests/models/postprocess_old_style_wrong_returns.py', Namespace(), [], orig_data)
        try:
            postprocess(samples, orig_data, feature_names)
        except ModelException as e:
            if e.title.find('postprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in postprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in postprocess")

    #### TESTS FOR PREPROCESS LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_with_broken_preprocess(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_broken.py', Namespace(), [], orig_data)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if isinstance(e.base, KeyError) and e.title.find('preprocessing data'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in preprocess, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for error in preprocess")

    def test_load_numpyro_model_old_style_preprocess(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_old_style.py', Namespace(), [], orig_data)
        train_data, num_data, feature_names = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))
        self.assertEqual(['new_first', 'new_second'], feature_names)

    def test_load_numpyro_model_preprocess_single_return(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_single_return.py', Namespace(), [], orig_data)
        train_data, num_data, feature_names = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))
        self.assertEqual(['new_first', 'new_second'], feature_names)

    def test_load_numpyro_model_preprocess_single_return_series(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_single_return_series.py', Namespace(), [], orig_data)
        train_data, num_data, feature_names = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(1, len(train_data))
        self.assertIsInstance(train_data[0], pd.Series)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]))
        self.assertEqual(['new_first'], feature_names)

    def test_load_numpyro_model_preprocess(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess.py', Namespace(), [], orig_data)
        train_data, num_data, feature_names = preprocess(orig_data)
        self.assertEqual(10, num_data)
        self.assertIsInstance(train_data, tuple)
        self.assertEqual(2, len(train_data))
        self.assertIsInstance(train_data[0], pd.DataFrame)
        self.assertIsInstance(train_data[1], pd.Series)
        self.assertTrue(np.allclose(orig_data['first'] + 2, train_data[0]['new_first']))
        self.assertTrue(np.allclose(orig_data['second'], train_data[0]['new_second']))
        self.assertTrue(np.allclose(orig_data['first'], train_data[1]))
        self.assertEqual(['new_first', 'new_second', 'y2'], feature_names)

    def test_load_numpyro_model_with_preprocess_wrong_returns(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_wrong_returns.py', Namespace(), [], orig_data)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong return value in preprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong return value in preprocess")

    def test_load_numpyro_model_with_preprocess_wrong_signature(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_wrong_signature.py', Namespace(), [], orig_data)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in preprocess, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for wrong signature in preprocess")

    def test_load_numpyro_model_with_preprocess_returns_array(self):
        orig_data = pd.DataFrame({'first': np.ones(10), 'second': np.zeros(10)})
        _, _, preprocess, _ = load_custom_numpyro_model('./tests/models/preprocess_returns_array.py', Namespace(), [], orig_data)
        try:
            preprocess(orig_data)
        except ModelException as e:
            if e.title.find('preprocessing data'.upper()) != -1 and e.msg.find('must return') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for non-dataframe returns, but did not give expected explanation; got:\n{e.format_message('')}")
        self.fail("load_custom_numpyro_model did not raise for non-dataframe returns in preprocess")

    #### TESTING MODEL LOADING AND ERROR WRAPPING
    def test_load_numpyro_model_simple_working_model(self):
        """ only verifies that no errors occur and all returned functions are not None """
        model, guide, preprocess, postprocess = load_custom_numpyro_model('./tests/models/simple_gauss_model.py', Namespace(), [], pd.DataFrame())
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
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_broken.py', Namespace(), [], pd.DataFrame())
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z)
        except ModelException as e:
            if isinstance(e.base, NameError) and e.title.find('model'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_without_num_obs_total(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_no_num_obs_total.py', Namespace(), [], pd.DataFrame())
        z = np.ones((10, 2))
        try:
            seed(model, jax.random.PRNGKey(0))(z, num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('num_obs_total') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_not_allowing_None_arguments(self):
        model, _, _, _ = load_custom_numpyro_model('./tests/models/simple_gauss_model_no_none.py', Namespace(), [], pd.DataFrame())
        try:
            seed(model, jax.random.PRNGKey(0))(num_obs_total=100)
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('None for synthesising data') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model")

    def test_load_numpyro_model_model_not_a_function(self):
        try:
            load_custom_numpyro_model('./tests/models/model_not_a_function.py', Namespace(), [], pd.DataFrame())
        except ModelException as e:
            if e.title.find('model'.upper()) != -1 and e.msg.find('must be a function') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for model not being a function, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for model not being a function")


    #### TESTING MODEL FACTORY
    def test_load_numpyro_model_model_factory(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/models/model_factory.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
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
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/models/model_factory_with_guide.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
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
        model, guide, preprocess, postprocess = load_custom_numpyro_model(
            './tests/models/model_factory_with_autoguide.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
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
        try:
            load_custom_numpyro_model(
                './tests/models/model_factory_broken.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
            )
        except ModelException as e:
            print(e.title)
            if e.title.find('model factory'.upper()) != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for error in model_factory, but did not correctly pass causal exception; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for error in model_factory")

    def test_load_numpyro_model_model_factory_wrong_signature(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            load_custom_numpyro_model(
                './tests/models/model_factory_wrong_signature.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('as argument') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong signature in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong signature in model_factory")

    def test_load_numpyro_model_model_factory_wrong_returns_none(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            load_custom_numpyro_model(
                './tests/models/model_factory_wrong_returns_none.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('either a model function or a tuple') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong returns in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong returns in model_factory")

    def test_load_numpyro_model_model_factory_wrong_returns_bad_tuple(self):
        orig_data = pd.DataFrame({'first': np.zeros(10), 'second': np.ones(10)})
        try:
            load_custom_numpyro_model(
                './tests/models/model_factory_wrong_returns_bad_tuple.py', Namespace(epsilon=1.), ['--prior_mu', '10'], orig_data
            )
        except ModelException as e:
            if e.title.find('model factory'.upper()) != -1 and e.msg.find('either a model function or a tuple') != -1:
                return
            self.fail(f"load_custom_numpyro_model did raise for wrong returns in model_factory, but did not give expected explanation; got: {e.format_message('')}")
        self.fail(f"load_custom_numpyro_model did not raise for wrong returns in model_factory")


    # TODO: test handling of guides
    # TODO: some integrated tests to ensure preprocess-model-postprocess pipeline error are handled well?