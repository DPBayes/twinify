#!/usr/bin/env python

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

"""
Twinify tools main script.
"""

import argparse
import twinify.cli.tools.check_model
from twinify import __version__

parser = argparse.ArgumentParser(description='Twinify tools: Various twinify utility scripts.')
parser.add_argument("--version", action='version', version=__version__)
subparsers = parser.add_subparsers(dest='command')
twinify.cli.tools.check_model.setup_argument_parser(subparsers.add_parser('check-model', help='Checks whether a given model works with twinify.'))

def main():
    args, unknown_args = parser.parse_known_args()

    if args.command =='check-model':
        return twinify.cli.tools.check_model.main(args, unknown_args)
    else:
        print('#### UNKNOWN OPERATING MODE ####')
        print(f'## twinify-tools does not know the command {args.command}')
        return 1

if __name__ == "__main__":
    exit(main())
