"""Tests for the Vasicek Model"""
import unittest
import numpy as np

from pyesg import CoxIngersollRoss, GeometricBrownianMotion, Vasicek


class TestVasicek(unittest.TestCase):
    """Test Vasicek Model"""

    def test_sample_shape(self):
        """Ensure the sample has the correct shape"""
        model = Vasicek()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(0.03, 1000, 30, 12, random_state=None)
        self.assertEqual(samples.shape, (1000, 30 * 12 + 1))

    def test_sample_first_value1(self):
        """Ensure the first scenario has the init value if init is a float"""
        model = Vasicek()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = 0.03
        samples = model.sample(init, 1000, 30, 12, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [0.03] * 1000)

    def test_sample_first_value2(self):
        """Ensure the first scenario has the init value if init is an array"""
        model = Vasicek()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = np.array([0.03])
        samples = model.sample(init, 1000, 30, 12, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [0.03] * 1000)

    def test_raises_value_error(self):
        """Ensure we raise a ValueError if init shape doesn't match n_scen"""
        model = Vasicek()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = np.array([0.03, 0.03])
        self.assertRaises(ValueError, model.sample, init, 1000, 30, 12)


class TestCoxIngersollRoss(unittest.TestCase):
    """Test Cox-Ingersoll-Ross Model"""

    def test_sample_shape(self):
        """Ensure the sample has the correct shape"""
        model = CoxIngersollRoss()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(0.03, 1000, 30, 12, random_state=None)
        self.assertEqual(samples.shape, (1000, 30 * 12 + 1))

    def test_sample_first_value1(self):
        """Ensure the first scenario has the init value if init is a float"""
        model = CoxIngersollRoss()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = 0.03
        samples = model.sample(init, 1000, 30, 12, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [0.03] * 1000)

    def test_sample_first_value2(self):
        """Ensure the first scenario has the init value if init is an array"""
        model = CoxIngersollRoss()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = np.array([0.03])
        samples = model.sample(init, 1000, 30, 12, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [0.03] * 1000)

    def test_raises_value_error(self):
        """Ensure we raise a ValueError if init shape doesn't match n_scen"""
        model = CoxIngersollRoss()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        init = np.array([0.03, 0.03])
        self.assertRaises(ValueError, model.sample, init, 1000, 30, 12)


class TestGeometricBrownianMotion(unittest.TestCase):
    """Test Cox-Ingersoll-Ross Model"""

    def test_sample_shape(self):
        """Ensure the sample has the correct shape"""
        model = GeometricBrownianMotion(n_indices=3)
        model.mu, model.sigma = 0.05, 0.20
        model.correlation = np.array(
            [[1.0, -0.19197, 0.0], [-0.19197, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        samples = model.sample(0.03, 1000, 30, 12, random_state=None)
        self.assertEqual(samples.shape, (1000, 30 * 12 + 1, 3))

    def test_coef_array(self):
        """Ensure the sample has the correct shape if coefs are passed as arrays"""
        model = GeometricBrownianMotion(n_indices=3)
        model.mu, model.sigma = np.full(3, 0.05), np.full(3, 0.20)
        model.correlation = np.array(
            [[1.0, -0.19197, 0.0], [-0.19197, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        samples = model.sample(0.03, 1000, 30, 12, random_state=None)
        self.assertEqual(samples.shape, (1000, 30 * 12 + 1, 3))