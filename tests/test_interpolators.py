"""Tests for Estimator Classes"""
import unittest
import numpy as np

from pyesg import NelsonSiegel, NelsonSiegelSvensson


class TestNelsonSiegel(unittest.TestCase):
    """Test Nelson Siegel Interpolator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = np.array([1, 2, 5, 10, 20, 30])
        self.y = np.array([0.003, 0.007, 0.012, 0.019, 0.028, 0.032])

    def test_predict_before_fit_error(self):
        """Throw an exception if we try to predict before fitting"""
        estimator = NelsonSiegel()
        self.assertRaises(RuntimeError, estimator.predict, self.X)

    def test_fixed_tau(self):
        """Ensure we don't overwrite tau if it's provided in the class constructor"""
        tau = 0.15
        estimator = NelsonSiegel(tau=tau)
        estimator.fit(self.X, self.y)
        self.assertEqual(estimator.tau, tau)

    def test_fit(self):
        """Ensure fitting this model sets the correct attributes"""
        estimator = NelsonSiegel()
        estimator.fit(self.X, self.y)
        self.assertTrue(hasattr(estimator, "beta0"))
        self.assertTrue(hasattr(estimator, "beta1"))
        self.assertTrue(hasattr(estimator, "beta2"))
        self.assertTrue(hasattr(estimator, "tau"))


class TestNelsonSiegelSvensson(unittest.TestCase):
    """Test Nelson Siegel Interpolator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = np.array([1, 2, 5, 10, 20, 30])
        self.y = np.array([0.003, 0.007, 0.012, 0.019, 0.028, 0.032])

    def test_predict_before_fit_error(self):
        """Throw an exception if we try to predict before fitting"""
        estimator = NelsonSiegelSvensson()
        self.assertRaises(RuntimeError, estimator.predict, self.X)

    def test_fixed_tau(self):
        """Ensure we don't overwrite tau if it's provided in the class constructor"""
        tau0, tau1 = 0.15, 0.3
        estimator = NelsonSiegelSvensson(tau0=tau0, tau1=tau1)
        estimator.fit(self.X, self.y)
        self.assertEqual(estimator.tau0, tau0)
        self.assertEqual(estimator.tau1, tau1)

    def test_fit(self):
        """Ensure fitting this model sets the correct attributes"""
        estimator = NelsonSiegelSvensson()
        estimator.fit(self.X, self.y)
        self.assertTrue(hasattr(estimator, "beta0"))
        self.assertTrue(hasattr(estimator, "beta1"))
        self.assertTrue(hasattr(estimator, "beta2"))
        self.assertTrue(hasattr(estimator, "beta3"))
        self.assertTrue(hasattr(estimator, "tau0"))
        self.assertTrue(hasattr(estimator, "tau1"))
