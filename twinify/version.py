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

MAJOR_VERSION = 2
MINOR_VERSION = 0
PATCH_VERSION = 0
EXT_VERSION = ""

EXT_VERSION_SUFFIX = f"-{EXT_VERSION}" if len(EXT_VERSION) > 0 else ""

VERSION = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}{EXT_VERSION_SUFFIX}"
