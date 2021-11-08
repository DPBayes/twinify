from os import close
import unittest
import subprocess
from typing import Tuple
import pandas as pd
import numpy as np

class CheckModelTests(unittest.TestCase):

    @staticmethod
    def run_check_model(model_file: str, data_file: str = 'gauss_data.csv') -> Tuple[int, str]:
        run_result = subprocess.run(
            ['twinify-tools', 'check-model', './tests/models/' + data_file, './tests/models/' + model_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output = run_result.stdout.decode('utf8')
        return run_result.returncode, output

    def verify_output(self, output: str, *args) -> bool:
        try:
            for s in args:
                self.assertNotEqual(output.find(s), -1, f"output does not contain '{s}'; got:\n{output}")
        except AssertionError:
            print(output)
            raise

    def test_model_not_found(self):
        retcode, output = self.run_check_model('does_not_exist.py')
        self.verify_output(output, "MODEL FILE")
        self.assertNotEqual(retcode, 0)

    def test_data_not_found(self):
        retcode, output = self.run_check_model('simple_gauss_model.py', data_file='does_not_exist')
        self.verify_output(output, "DATA FILE")
        self.assertNotEqual(retcode, 0)

    def test_postprocess_broken(self):
        retcode, output = self.run_check_model('postprocess_broken.py')
        self.verify_output(output, "POSTPROCESSING", "KeyError")
        self.assertNotEqual(retcode, 0)

    def test_postprocess_old_style_broken(self):
        retcode, output = self.run_check_model('postprocess_broken.py')
        self.verify_output(output, "POSTPROCESSING", "KeyError")
        self.assertNotEqual(retcode, 0)

    def test_postprocess_old_style_wrong_returns(self):
        retcode, output = self.run_check_model('postprocess_old_style_wrong_returns.py')
        self.verify_output(output, "POSTPROCESSING", "must return")
        self.assertNotEqual(retcode, 0)

    def test_postprocess_wrong_returns(self):
        retcode, output = self.run_check_model('postprocess_wrong_returns.py')
        self.verify_output(output, "POSTPROCESSING", "must return")
        self.assertNotEqual(retcode, 0)

    def test_postprocess_wrong_signature(self):
        retcode, output = self.run_check_model('postprocess_wrong_signature.py')
        self.verify_output(output, "POSTPROCESSING", "as argument")
        self.assertNotEqual(retcode, 0)

    def test_preprocess_broken(self):
        retcode, output = self.run_check_model('preprocess_broken.py')
        self.verify_output(output, "PREPROCESSING", "KeyError")
        self.assertNotEqual(retcode, 0)

    def test_preprocess_wrong_returns(self):
        retcode, output = self.run_check_model('preprocess_wrong_returns.py')
        self.verify_output(output, "PREPROCESSING", "must return")
        self.assertNotEqual(retcode, 0)

    def test_preprocess_wrong_signature(self):
        retcode, output = self.run_check_model('preprocess_wrong_signature.py')
        self.verify_output(output, "PREPROCESSING", "as argument")
        self.assertNotEqual(retcode, 0)

    def test_preprocess_returns_array(self):
        retcode, output = self.run_check_model('preprocess_returns_array.py')
        self.verify_output(output, "PREPROCESSING", "must return")
        self.assertNotEqual(retcode, 0)

    def test_broken_model(self):
        retcode, output = self.run_check_model('simple_gauss_model_broken.py')
        self.verify_output(output, "MODEL", "NameError")
        self.assertNotEqual(retcode, 0)

    def test_no_none_model(self):
        retcode, output = self.run_check_model('simple_gauss_model_no_none.py')
        self.verify_output(output, "MODEL", "None for synthesising data")
        self.assertNotEqual(retcode, 0)

    def test_no_num_obs_total_model(self):
        retcode, output = self.run_check_model('simple_gauss_model_no_num_obs_total.py')
        self.verify_output(output, "MODEL", "num_obs_total")
        self.assertNotEqual(retcode, 0)

    def test_syntax_error(self):
        retcode, output = self.run_check_model('syntax_error.py')
        self.verify_output(output, "PARSE", "SyntaxError")
        self.assertNotEqual(retcode, 0)

    def test_new_model_old_postprocess(self):
        retcode, output = self.run_check_model('new_model_old_postprocess.py')
        self.verify_output(output, "POSTPROCESSING", "postprocessing function with a single argument", "'x'")
        self.assertNotEqual(retcode, 0)

    def test_simple_working_model(self):
        retcode, output = self.run_check_model('simple_gauss_model.py')
        self.verify_output(output, "okay")
        self.assertEqual(retcode, 0)

    def test_model_not_a_function(self):
        retcode, output = self.run_check_model('model_not_a_function.py')
        self.verify_output(output, "MODEL", "PARSE", "must be a function")
        self.assertNotEqual(retcode, 0)

    def test_model_factory(self):
        retcode, output = self.run_check_model('model_factory.py')
        self.verify_output(output, "okay")
        self.assertEqual(retcode, 0)

    def test_model_factory_broken(self):
        retcode, output = self.run_check_model('model_factory_broken.py')
        self.verify_output(output, "FACTORY", "AttributeError", "unspecified_arg")
        self.assertNotEqual(retcode, 0)

    def test_model_factory_wrong_signature(self):
        retcode, output = self.run_check_model('model_factory_wrong_signature.py')
        self.verify_output(output, "FACTORY", "as argument")
        self.assertNotEqual(retcode, 0)

    def test_model_factory_wrong_returns(self):
        retcode, output = self.run_check_model('model_factory_wrong_returns.py')
        self.verify_output(output, "MODEL", "must be a function")
        self.assertNotEqual(retcode, 0)
