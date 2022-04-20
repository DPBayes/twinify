# Copyright 2022 twinify Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import namedtuple
import pickle
from twinify.version import VERSION


TwinifyRunResult = namedtuple('TwinifyRunResult',
    ('model_params', 'elbo', 'twinify_args', 'unknown_args', 'twinify_version')
)

def store_twinify_run_result(file_or_path, model_params, elbo, twinify_args, unknown_args):
    """ Stores the results of a twinify run to a file in a binary pickle format.

    Args:
        file_or_path (FileIO or str): A file descriptor or a string containg a file path to write the results to.
        model_params: The parameters for the model inferred during the twinify run.
        elbo: The final value of the ELBO after completing the inference.
        twinify_args (argparse.Namespace): The namespace object containing all command line arguments parsed by twinify.
        unknown_args (list of str): Remaining command line arguments that were not parsed/ignored by twinify.
    """
    result = TwinifyRunResult(model_params, elbo, twinify_args, unknown_args, VERSION)

    if isinstance(file_or_path, str):
        with open(file_or_path, "wb") as f:
            pickle.dump(result, f)
    else:
        pickle.dump(result, file_or_path)


def load_twinify_run_result(file_or_path) -> TwinifyRunResult:
    """ Loads the results of a twinify run from a binary pickle file.

    Args:
        file_or_path (FileIO or str): A file descriptor or a string containg a file path to read the results from.
    Returns:
        A TwinifyRunResult object containing the inferred model parameters, final ELBO value, command line arguments
        given to twinify for the corresponding run and the version of twinify.
    """
    def load_from_file(f):
        result = pickle.load(f)
        assert isinstance(result, TwinifyRunResult)
        return result

    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as f:
            return load_from_file(f)
    else:
        return load_from_file(file_or_path)
