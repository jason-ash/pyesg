"""Test functions from utils.py"""
import unittest
import numpy as np

from pyesg.utils import _has_valid_cholesky, check_random_state, to_array


class TestUtils(unittest.TestCase):
    """Test functions from utils.py"""

    def test_valid_matrix_passes_cholesky(self):
        """Ensure a valid matrix passes"""
        matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        self.assertTrue(_has_valid_cholesky(matrix))

    def test_invalid_matrix_fails_cholesky(self):
        """Ensure an invalid matrix fails"""
        matrix = np.array([[1.0, 1.0, 0.6], [1.0, 1.0, 0.6], [0.6, 0.6, 1.0]])
        self.assertFalse(_has_valid_cholesky(matrix))

    def test_invalid_random_state_raises(self):
        """Ensure we raise a ValueError for an invalid random_seed arguments"""
        seed = "random_seed"
        self.assertRaises(ValueError, check_random_state, seed)

    def test_invalid_arraylike_raises(self):
        """Ensure we raise a TypeError if we can't convert an arg to an array"""
        value = {"array": [1, 2, 3, 4, 5]}
        self.assertRaises(TypeError, to_array, value)
