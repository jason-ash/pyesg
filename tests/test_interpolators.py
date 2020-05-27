"""Tests for Estimator Classes"""
import unittest
import numpy as np

from pyesg import NelsonSiegelInterpolator, SvenssonInterpolator


class TestNelsonSiegelInterpolator(unittest.TestCase):
    """Test Nelson Siegel Interpolator"""

    @classmethod
    def setUpClass(cls):
        cls.X = np.array([1, 2, 5, 10, 20, 30])
        cls.y = np.array([0.003, 0.007, 0.012, 0.019, 0.028, 0.032])

    def test_repr(self):
        """Ensure we have the right repr for the models"""
        estimator = NelsonSiegelInterpolator()
        output = "<pyesg.NelsonSiegelInterpolator>"
        self.assertEqual(estimator.__repr__(), output)

    def test_predict_before_fit_error(self):
        """Throw an exception if we try to predict before fitting"""
        estimator = NelsonSiegelInterpolator()
        self.assertRaises(RuntimeError, estimator.predict, self.X)

    def test_fixed_tau(self):
        """Ensure we don't overwrite tau if it's provided in the class constructor"""
        tau = 0.15
        estimator = NelsonSiegelInterpolator(tau=tau)
        estimator.fit(self.X, self.y)
        self.assertEqual(estimator.tau, tau)

    def test_fit(self):
        """Ensure fitting this model sets the correct attributes"""
        estimator = NelsonSiegelInterpolator()
        estimator.fit(self.X, self.y)
        self.assertTrue(hasattr(estimator, "beta0"))
        self.assertTrue(hasattr(estimator, "beta1"))
        self.assertTrue(hasattr(estimator, "beta2"))
        self.assertTrue(hasattr(estimator, "tau"))

    def test_predict(self):
        """Test the model predict function"""
        estimator = NelsonSiegelInterpolator()
        estimator.fit(self.X, self.y)
        expected = estimator.predict(np.array([3, 7, 15]))
        actual = np.array([0.008234350407870, 0.01532923389080558, 0.02428871544904800])
        self.assertIsNone(np.testing.assert_array_almost_equal(expected, actual))


class TestSvenssonInterpolator(unittest.TestCase):
    """Test Nelson Siegel Interpolator"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = np.array([1, 2, 5, 10, 20, 30])
        self.y = np.array([0.003, 0.007, 0.012, 0.019, 0.028, 0.032])

    def test_predict_before_fit_error(self):
        """Throw an exception if we try to predict before fitting"""
        estimator = SvenssonInterpolator()
        self.assertRaises(RuntimeError, estimator.predict, self.X)

    def test_fixed_tau(self):
        """Ensure we don't overwrite tau if it's provided in the class constructor"""
        tau0, tau1 = 0.15, 0.3
        estimator = SvenssonInterpolator(tau0=tau0, tau1=tau1)
        estimator.fit(self.X, self.y)
        self.assertEqual(estimator.tau0, tau0)
        self.assertEqual(estimator.tau1, tau1)

    def test_fit(self):
        """Ensure fitting this model sets the correct attributes"""
        estimator = SvenssonInterpolator()
        estimator.fit(self.X, self.y)
        self.assertTrue(hasattr(estimator, "beta0"))
        self.assertTrue(hasattr(estimator, "beta1"))
        self.assertTrue(hasattr(estimator, "beta2"))
        self.assertTrue(hasattr(estimator, "beta3"))
        self.assertTrue(hasattr(estimator, "tau0"))
        self.assertTrue(hasattr(estimator, "tau1"))
