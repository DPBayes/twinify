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

import os
from pathlib import Path
from typing import Optional
import shutil

TEST_DIRECTORY_PATH = '/tmp/napsu_test'


# TODO: Create OS independent implementation
def create_test_directory(directory: Optional[str] = TEST_DIRECTORY_PATH, exists: Optional[str] = 'ignore') -> None:
    """Create shared directory for IO tests.
    Args:
        directory (string): Path to test directory.
        exists (string, 'ignore', 'raise'):
            If directory already exists and argument was 'ignore', don't do anything.
            If directory exists and argument was 'raise', raise an Exception.
    """

    raise_exists = exists == 'raise'

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=raise_exists)


def file_exists(path_to_file: str) -> bool:
    """Check if file exists in path, returns False for folders.
    Args:
        path_to_file (str): Path that should contain file.
    Return:
        bool: If file exists and contains file.
    """
    path = Path(path_to_file)
    return path.exists() and path.is_file()


def purge_test_directory(directory: Optional[str] = TEST_DIRECTORY_PATH) -> None:
    """Remove all files from test directory and remove the directory itself.
    Args:
        directory (string): Path to test directory.
    """
    shutil.rmtree(directory, ignore_errors=False)
