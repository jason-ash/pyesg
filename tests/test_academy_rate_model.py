"""Test the AcademyRateModel class and its related functions"""
import unittest
import numpy as np

from pyesg import AcademyRateProcess
from pyesg.academy_rate_model import (
    interpolate,
    perturb,
    scenario_rank,
    scenario_significance_value,
    scenario_subset,
    AcademyRateModel,
)
from pyesg.datasets import load_academy_sample_scenario


class TestAcademyRateModel(unittest.TestCase):
    """Test the AcademyRateModel class"""

    @classmethod
    def setUpClass(cls):
        cls.model = AcademyRateModel()
        cls.test_scenario = load_academy_sample_scenario()

    def test_interpolate(self):
        """Ensure the interpolate function works as expected"""
        # the function expects multiple short rates and long rates
        short_rate = np.full((1, 1), 0.01)
        long_rate = np.full((1, 1), 0.03)
        maturities = np.array([0.5, 1.0, 5.0, 10.0, 20.0, 30.0])
        actual = interpolate(short_rate, long_rate, interpolated_maturities=maturities)
        expected = np.array([[[0.00765, 0.01, 0.021208, 0.026554, 0.03, 0.031191]]])
        self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected))

    def test_perturb(self):
        """Ensure the pertubtation function works as expected"""
        # this function essentially grades the scenario from the starting yield curve up
        # to the scenario value over the first projection year of the scenarios. We'll
        # test this by making sure this happens as expected for some dummy scenarios.
        yield_curve = np.array([0.005, 0.015, 0.025, 0.035, 0.045])
        scenarios = np.array([[[0.01, 0.02, 0.03, 0.04, 0.05]] * 3])
        actual = perturb(scenarios=scenarios, n_steps=2, yield_curve=yield_curve)
        expected = [
            [0.005, 0.015, 0.025, 0.035, 0.045],
            [0.0075, 0.0175, 0.0275, 0.0375, 0.0475],
            [0.01, 0.02, 0.03, 0.04, 0.05],
        ]
        expected = np.array([expected])
        self.assertIsNone(np.testing.assert_array_almost_equal(actual, expected))

    def test_scenario_shape(self):
        """Ensure the scenarios method produces the right shape"""
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=10, n_steps=30)
        self.assertEqual(scenarios.shape, (10, 31, 10))

    def test_scenario_significance_shape(self):
        """Ensure the scenario significance array has the right shape"""
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=10, n_steps=30)
        significance = scenario_significance_value(scenarios)
        self.assertEqual((10,), significance.shape)

    def test_scenario_rank_shape(self):
        """Ensure the scenario rank array has the right shape"""
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=10, n_steps=30)
        rank = scenario_rank(scenarios)
        self.assertEqual((10,), rank.shape)

    def test_scenario_values(self):
        """Compare the pyesg model vs. a single scenario from the AAA Excel model"""
        model = AcademyRateModel(volatility=self.test_scenario["volatility"])
        model.yield_curve = self.test_scenario["yield_curve"]
        model.process = AcademyRateProcess(**self.test_scenario["process_parameters"])
        scenario = model.scenarios(
            dt=self.test_scenario["dt"],
            n_scenarios=self.test_scenario["n_scenarios"],
            n_steps=self.test_scenario["n_steps"],
            floor=self.test_scenario["floor"],
            random_state=self.test_scenario["random_state"],
        )
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                self.test_scenario["sample_scenario"], scenario[0]
            )
        )

    def test_scenario_significance_value(self):
        """Ensure the scenario significance value matches what we expect"""
        model = AcademyRateModel(volatility=self.test_scenario["volatility"])
        model.yield_curve = self.test_scenario["yield_curve"]
        model.process = AcademyRateProcess(**self.test_scenario["process_parameters"])
        scenario = model.scenarios(
            dt=self.test_scenario["dt"],
            n_scenarios=self.test_scenario["n_scenarios"],
            n_steps=self.test_scenario["n_steps"],
            floor=self.test_scenario["floor"],
            random_state=self.test_scenario["random_state"],
        )
        significance = scenario_significance_value(scenario)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                self.test_scenario["sample_scenario_significance_value"], significance
            )
        )

    def test_scenario_subset_shape(self):
        """Ensure the scenario subset has the right shape"""
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=100, n_steps=30)
        subset = scenario_subset(scenarios, 50)
        self.assertEqual(50, len(subset))

    def test_scenario_subset_ranks(self):
        """Ensure the subset of scenarios has the right rankings"""
        # we'll sample 50 scenarios from a batch of 100, expecting every odd numbered
        # rank, like 1, 3, 5, ... 99.
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=100, n_steps=30)
        ranks = scenario_rank(scenarios)
        expected = scenarios[ranks[np.arange(1, 100, 2)], :, :]
        actual = scenario_subset(scenarios, 50)
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_scenario_subset_error(self):
        """Ensure we raise an error if we can't return a subset of scenarios"""
        scenarios = self.model.scenarios(dt=1 / 12, n_scenarios=100, n_steps=30)
        self.assertRaises(RuntimeError, scenario_subset, scenarios, 47)
        self.assertRaises(RuntimeError, scenario_subset, scenarios, 150)
