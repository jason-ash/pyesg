"""Tests for the Vasicek Model"""
import doctest
import unittest
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
import numpy as np

from pyesg import models
from pyesg import WienerProcess, OrnsteinUhlenbeckProcess, GeometricBrownianMotion


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
    tests.addTests(doctest.DocTestSuite(models))
    return tests


class TestWienerProcess(unittest.TestCase):
    """Test WienerProcess"""

    @settings(deadline=None)
    @given(n_scen=integers(1, 100), n_year=integers(1, 20), n_step=integers(1, 252))
    def test_sample_shapes(self, n_scen, n_year, n_step):
        """Ensure samples have the correct shape"""
        model = WienerProcess()
        model.mu, model.sigma = 0.045, 0.015
        samples = model.sample(0.03, n_scen, n_year, n_step, random_state=None)
        self.assertEqual(samples.shape, (n_scen, 1 + n_year * n_step))

    @given(init=floats(0, allow_infinity=False))
    def test_sample_init_value1(self, init):
        """Ensure the init value is used as the scenario start value"""
        model = WienerProcess()
        model.mu, model.sigma = 0.045, 0.015
        samples = model.sample(init, 10, 10, 1, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [init] * 10)

    @given(init=floats(0, allow_infinity=False))
    def test_sample_init_value2(self, init):
        """Ensure the init value is used as the scenario start value"""
        model = WienerProcess()
        model.mu, model.sigma = 0.045, 0.015
        samples = model.sample(np.full(10, init), 10, 10, 1, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [init] * 10)

    def test_raises_value_error(self):
        """Ensure we raise a ValueError if init shape doesn't match n_scen"""
        model = WienerProcess()
        model.mu, model.sigma = 0.045, 0.015
        init = np.array([0.03, 0.03])
        self.assertRaises(ValueError, model.sample, init, 1000, 30, 12)


class TestOrnsteinUhlenbeckProcess(unittest.TestCase):
    """Test OrnsteinUhlenbeckProcess"""

    @settings(deadline=None)
    @given(n_scen=integers(1, 100), n_year=integers(1, 20), n_step=integers(1, 252))
    def test_sample_shapes(self, n_scen, n_year, n_step):
        """Ensure samples have the correct shape"""
        model = OrnsteinUhlenbeckProcess()
        model.theta, model.mu, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(0.03, n_scen, n_year, n_step, random_state=None)
        self.assertEqual(samples.shape, (n_scen, 1 + n_year * n_step))

    @given(init=floats(0, allow_infinity=False))
    def test_sample_init_value1(self, init):
        """Ensure the init value is used as the scenario start value"""
        model = OrnsteinUhlenbeckProcess()
        model.theta, model.mu, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(init, 10, 10, 1, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [init] * 10)

    @given(init=floats(0, allow_infinity=False))
    def test_sample_init_value2(self, init):
        """Ensure the init value is used as the scenario start value"""
        model = OrnsteinUhlenbeckProcess()
        model.theta, model.mu, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(np.full(10, init), 10, 10, 1, random_state=None)
        self.assertListEqual(list(samples[:, 0]), [init] * 10)

    def test_raises_value_error(self):
        """Ensure we raise a ValueError if init shape doesn't match n_scen"""
        model = OrnsteinUhlenbeckProcess()
        model.theta, model.mu, model.sigma = 0.15, 0.045, 0.015
        init = np.array([0.03, 0.03])
        self.assertRaises(ValueError, model.sample, init, 1000, 30, 12)


class TestGeometricBrownianMotion(unittest.TestCase):
    """Test GeometricBrownianMotion"""

    @settings(deadline=None)
    @given(n_scen=integers(1, 100), n_year=integers(1, 20), n_step=integers(1, 252))
    def test_sample_shapes(self, n_scen, n_year, n_step):
        """Ensure samples have the correct shape"""
        model = GeometricBrownianMotion()
        model.mu, model.sigma = 0.05, 0.20
        samples = model.sample(100.0, n_scen, n_year, n_step, random_state=None)
        self.assertEqual(samples.shape, (n_scen, 1 + n_year * n_step))
