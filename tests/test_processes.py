"""Tests for stochastic processes"""
import unittest
import numpy as np

from pyesg import AcademyRateProcess, WienerProcess


class TestWienerProcess(unittest.TestCase):
    """Test WienerProcess"""

    @classmethod
    def setUpClass(cls):
        cls.model = WienerProcess(mu=0.045, sigma=0.15)

    def test_step_shapes(self):
        """Ensure single step samples have the correct shape"""
        steps = self.model.step(x0=0.05, dt=1.0)
        self.assertEqual(steps.shape, (1,))
        steps = self.model.step(x0=np.array(0.05), dt=1.0)
        self.assertEqual(steps.shape, (1,))
        steps = self.model.step(x0=np.array([0.05]), dt=1.0)
        self.assertEqual(steps.shape, (1,))

    def test_scenario_shapes(self):
        """Ensure scenario method produces the correct shape"""
        # generate 100 scenarios of 30 timesteps from a single start values
        scenarios = self.model.scenarios(x0=0.05, dt=1.0, shape=(100, 30))
        self.assertEqual(scenarios.shape, (100, 31))

        # generate 100 scenarios of 30 timesteps from an array of 100 start values
        scenarios = self.model.scenarios(x0=np.full(100, 0.05), dt=1.0, shape=(100, 30))
        self.assertEqual(scenarios.shape, (100, 31))

        # raise an error if we can't match the start values with the number of scenarios
        # try to make a 5-scenario/5-timestep array from a starting list of three values
        self.assertRaises(
            ValueError, self.model.scenarios, [0.02, 0.03, 0.04], dt=1.0, shape=(5, 5)
        )


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
