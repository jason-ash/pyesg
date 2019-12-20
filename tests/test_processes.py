"""Tests for the Vasicek Model"""
import doctest
import unittest
import numpy as np

from pyesg import AcademyRateProcess, WienerProcess
from pyesg.processes import (
    academy_rate_process,
    black_scholes_process,
    cox_ingersoll_ross_process,
    geometric_brownian_motion,
    heston_process,
    ornstein_uhlenbeck_process,
    wiener_process,
)
from pyesg import utils


# pylint: disable=unused-argument,line-too-long
def load_tests(loader, tests, ignored):
    """
    This function allows unittest to discover doctests in the module.

    It appears not to use the arguments (or do anything, really), but this is
    used internally by unittest to "discover" the tests it needs to run.

    References
    ----------
    https://stackoverflow.com/questions/5681330/using-doctests-from-within-unittests
    https://docs.python.org/2/library/unittest.html#load-tests-protocol
    """
    tests.addTests(doctest.DocTestSuite(utils))
    tests.addTests(doctest.DocTestSuite(academy_rate_process))
    tests.addTests(doctest.DocTestSuite(black_scholes_process))
    tests.addTests(doctest.DocTestSuite(cox_ingersoll_ross_process))
    tests.addTests(doctest.DocTestSuite(geometric_brownian_motion))
    tests.addTests(doctest.DocTestSuite(heston_process))
    tests.addTests(doctest.DocTestSuite(ornstein_uhlenbeck_process))
    tests.addTests(doctest.DocTestSuite(wiener_process))
    return tests


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
