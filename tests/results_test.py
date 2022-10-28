import argparse
import unittest
import jax
import jax.numpy as jnp
from twinify.results import store_twinify_run_result, load_twinify_run_result
import tempfile
import twinify.version

class ResultIOTests(unittest.TestCase):

    def test_store_and_load_results(self):
        model_params = {
            'param1': jnp.ones((2, 3)),
            'param2': jnp.zeros((5,))
        }
        elbo = -77.2
        parser = argparse.ArgumentParser()
        parser.add_argument("--arg1", type=str)
        known_args, unknown_args = parser.parse_known_args(['--arg1', 'val1', '--unknown_arg', 'unknown_val'])
        with tempfile.TemporaryFile("w+b") as f:
            store_twinify_run_result(f, model_params, elbo, known_args, unknown_args)

            f.seek(0)
            loaded_result = load_twinify_run_result(f)

            self.assertTrue(
                jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.allclose, model_params, loaded_result.model_params))
            )
            self.assertAlmostEqual(elbo, loaded_result.elbo)
            self.assertEqual(known_args, loaded_result.twinify_args)
            self.assertSequenceEqual(unknown_args, loaded_result.unknown_args)
            self.assertEqual(twinify.version.VERSION, loaded_result.twinify_version)
