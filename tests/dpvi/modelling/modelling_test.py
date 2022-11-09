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
