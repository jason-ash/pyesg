"""Tests for stochastic processes"""
import unittest
import numpy as np
import pandas as pd

from pyesg import (
    # AcademyRateProcess,
    BlackScholesProcess,
    CoxIngersollRossProcess,
    GeometricBrownianMotion,
    # HestonProcess,
    # JointWienerProcess,
    OrnsteinUhlenbeckProcess,
    WienerProcess,
)


class BaseProcessMixin:
    """
    Holds common tests for all processes. Each model should subclass this mixin to
    inherit tests for scenario generation, expectation, dimension-checking, etc. Each
    model subclass will need to create a `self.model` instance that can be tested.
    """

    def test_single_initial_value_expectation_shape(self):
        """Ensure the expectation has the correct shape for a single start value"""
        exp = self.model.expectation(x0=self.single_x0, dt=1.0)
        self.assertEqual(exp.shape, self.single_x0.shape)

    def test_multiple_initial_value_expectation_shape(self):
        """Ensure the expectation has the correct shape for a single start value"""
        exp = self.model.expectation(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(exp.shape, self.multiple_x0.shape)

    def test_single_initial_value_standard_deviation_shape(self):
        """Ensure the std deviation has the correct shape for a single start value"""
        std = self.model.standard_deviation(x0=self.single_x0, dt=1.0)
        self.assertEqual(std.shape, self.single_x0.shape)

    def test_multiple_initial_value_standard_deviation_shape(self):
        """Ensure the std deviation has the correct shape for multiple start value"""
        std = self.model.standard_deviation(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(std.shape, self.multiple_x0.shape)

    def test_single_initial_value_step_shape(self):
        """Ensure the step function returns an array matching the initial array"""
        step = self.model.step(x0=self.single_x0, dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_multiple_initial_value_step_shape(self):
        """Ensure the step function returns an array matching the initial array"""
        step = self.model.step(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(step.shape, self.multiple_x0.shape)

    def test_single_initial_value_step_list_dtype(self):
        """Ensure we can pass single initial values as a list of floats"""
        step = self.model.step(x0=list(self.single_x0), dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_step_float_dtype(self):
        """Ensure we can pass single initial values as a float"""
        step = self.model.step(x0=float(self.single_x0), dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_step_series_dtype(self):
        """Ensure we can pass single initial values as a pd.Series"""
        step = self.model.step(x0=pd.Series(self.single_x0), dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_unique_scenarios(self):
        """Ensure when we generate lots of scenarios, they are all unique"""
        scenarios = self.model.scenarios(x0=self.single_x0, dt=1.0, shape=(50, 30))
        self.assertEqual(50, len(set(scenarios[:, -1])))

    def test_single_initial_value_scenario_shape(self):
        """Ensure scenarios from a single start value have the right shape"""
        scenarios = self.model.scenarios(x0=self.single_x0, dt=1.0, shape=(50, 30))
        self.assertEqual(scenarios.shape, (50, 31))

    def test_multiple_initial_value_scenario_shape(self):
        """Ensure scenarios from a single start value have the right shape"""
        scenarios = self.model.scenarios(
            x0=self.multiple_x0, dt=1.0, shape=(self.multiple_x0.shape[0], 30)
        )
        self.assertEqual(scenarios.shape, (self.multiple_x0.shape[0], 31))


class TestBlackScholesProcess(BaseProcessMixin, unittest.TestCase):
    """Test BlackScholesProcess"""

    @classmethod
    def setUpClass(cls):
        cls.model = BlackScholesProcess.example()
        cls.single_x0 = np.array([100.0])
        cls.multiple_x0 = np.full(50, 100.0)


class TestCoxIngersollRossProcess(BaseProcessMixin, unittest.TestCase):
    """Test CoxIngersollRossProcess"""

    @classmethod
    def setUpClass(cls):
        cls.model = CoxIngersollRossProcess.example()
        cls.single_x0 = np.array([0.03])
        cls.multiple_x0 = np.full(50, 0.03)


class TestGeometricBrownianMotion(BaseProcessMixin, unittest.TestCase):
    """Test GeometricBrownianMotion"""

    @classmethod
    def setUpClass(cls):
        cls.model = GeometricBrownianMotion.example()
        cls.single_x0 = np.array([100.0])
        cls.multiple_x0 = np.full(50, 100.0)


class TestOrnsteinUhlenbeckProcess(BaseProcessMixin, unittest.TestCase):
    """Test OrnsteinUhlenbeckProcess"""

    @classmethod
    def setUpClass(cls):
        cls.model = OrnsteinUhlenbeckProcess.example()
        cls.single_x0 = np.array([0.03])
        cls.multiple_x0 = np.full(50, 0.03)


class TestWienerProcess(BaseProcessMixin, unittest.TestCase):
    """Test WienerProcess"""

    @classmethod
    def setUpClass(cls):
        cls.model = WienerProcess.example()
        cls.single_x0 = np.array([100.0])
        cls.multiple_x0 = np.full(50, 100.0)


# class TestAcademyRateProcess(BaseProcessMixin, unittest.TestCase):
#     """Test AcademyRateProcess"""

#     @classmethod
#     def setUpClass(cls):
#         cls.model = AcademyRateProcess.example()
#         cls.single_x0 = np.array([0.03, 0.0024, 0.03])
#         cls.multiple_x0 = np.array([
#             [0.03, 0.0024, 0.03],
#             [0.03, 0.0024, 0.03],
#             [0.03, 0.0024, 0.03],
#             [0.03, 0.0024, 0.03],
#         ])

#     def test_cap(self):
#         """Ensure that the drift process produces values within the acceptable range"""
#         # ensure the expectation gets capped at the high end by long_rate_max
#         x0 = np.array([self.model.long_rate_max * 1.1, 0.0024, 0.03])
#         self.assertAlmostEqual(
#             self.model.expectation(x0=x0, dt=1.0)[0], self.model.long_rate_max
#         )

#         # ensure the expectation gets floored at the low end by long_rate_max
#         x0 = np.array([self.model.long_rate_min * 0.9, 0.0024, 0.03])
#         self.assertAlmostEqual(
#             self.model.expectation(x0=x0, dt=1.0)[0], self.model.long_rate_min
#         )


# class TestHestonProcess(BaseProcessMixin, unittest.TestCase):
#     """Test HestonProcess"""

#     @classmethod
#     def setUpClass(cls):
#         cls.model = HestonProcess.example()
#         cls.single_x0 = np.array([10., 0.04])
#         cls.multiple_x0 = np.array([
#             [10., 0.04],
#             [10., 0.04],
#             [10., 0.04],
#             [10., 0.04],
#         ])


# class TestJointWienerProcess(BaseProcessMixin, unittest.TestCase):
#     """Test JointWienerProcess"""

#     @classmethod
#     def setUpClass(cls):
#         cls.model = JointWienerProcess.example()
#         cls.single_x0 = np.array([100.0, 90.0])
#         cls.multiple_x0 = np.array([
#             [100.0, 90.0],
#             [100.0, 90.0],
#             [100.0, 90.0],
#             [100.0, 90.0],
#         ])
