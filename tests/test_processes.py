"""Tests for stochastic processes"""
import unittest
import numpy as np

from pyesg import AcademyRateProcess, WienerProcess


class TestWienerProcess(unittest.TestCase):
    """Test WienerProcess"""

    def test_sample_shapes(self):
        """Ensure samples have the correct shape"""
        model = WienerProcess(mu=0.045, sigma=0.15)
        steps = model.step(x0=0.05, dt=1.0, random_state=None)
        self.assertEqual(steps.shape, (1,))
        steps = model.step(x0=np.array(0.05), dt=1.0, random_state=None)
        self.assertEqual(steps.shape, (1,))
        steps = model.step(x0=np.array([0.05]), dt=1.0, random_state=None)
        self.assertEqual(steps.shape, (1,))


class TestAcademyRateProcess(unittest.TestCase):
    """Test AcademyRateProcess"""

    def test_cap(self):
        """Ensure that the drift process produces values within the acceptable range"""
        arp = AcademyRateProcess(long_rate_min=0.0115, long_rate_max=0.18)

        # ensure the expectation is never higher than the long_rate_max
        x0 = np.array([0.2, 0.0024, 0.03])
        self.assertLessEqual(arp.expectation(x0=x0, dt=1.0)[0], 0.18)

        # ensure the expectatin is never lower than the long_rate_min
        x0 = np.array([0.005, 0.0024, 0.03])
        self.assertGreaterEqual(arp.expectation(x0=x0, dt=1.0)[0], 0.0115)
