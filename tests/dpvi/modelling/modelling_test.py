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

import jax.numpy as jnp
import numpy as np

from twinify.dpvi.modelling import slice_feature

class ModellingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = jnp.array(np.ones((10, 3)) * np.array([0, 1, 2]))

    def test_slice_feature_single(self) -> None:
        feat = slice_feature(self.data, 1, 2)
        expected_feat = self.data[:, 1]

        self.assertTrue(np.all(expected_feat == feat))

    def test_slice_feature_single_None(self) -> None:
        feat = slice_feature(None, 1)
        self.assertIsNone(feat)

    def test_slice_feature_multi(self) -> None:
        feats = slice_feature(self.data, 1, 3)
        expected_feats = self.data[:, 1:3]

        self.assertTrue(np.all(expected_feats == feats))

    def test_slice_feature_multi_None(self) -> None:
        feats = slice_feature(None, 1, 3)
        self.assertIsNone(feats)

    def test_slice_feature_multi_strided(self) -> None:
        feats = slice_feature(self.data, 0, 3, 2)
        expected_feats = self.data[:, [0, 2]]

        self.assertTrue(np.all(expected_feats == feats))

    def test_slice_feature_multi_strided_None(self) -> None:
        feats = slice_feature(None, 0, 3, 2)
        self.assertIsNone(feats)

    def test_slice_feature_dtype(self) -> None:
        feats = slice_feature(self.data, 1, 3, dtype=jnp.int32)
        expected_feats = self.data[:, 1:3].astype(jnp.int32)

        self.assertTrue(np.all(expected_feats == feats))
