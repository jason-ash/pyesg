"""Tests for the Vasicek Model"""
import doctest
import unittest
import numpy as np

from pyesg import WienerProcess
from pyesg.processes import (
    academy_process,
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
    tests.addTests(doctest.DocTestSuite(academy_process))
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
        steps = model.step(x0=np.full(10, 0.05), dt=1.0, random_state=None)
        self.assertEqual(steps.shape, (10,))
