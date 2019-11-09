"""Tests for the Vasicek Model"""
from io import StringIO
import unittest
import numpy as np
import pandas as pd

from pyesg import Vasicek
from pyesg.datasets import load_ust_historical


class TestVasicek(unittest.TestCase):
    """Test Vasicek Model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_mle(self):
        """Test MLE on a small dataset with confirmed results"""
        data = """0, 3
            0.25, 1.76
            0.5, 1.2693
            0.75, 1.196
            1, 0.9468
            1.25, 0.9532
            1.5, 0.6252
            1.75, 0.8604
            2, 1.0984
            2.25, 1.431
            2.5, 1.3019
            2.75, 1.4005
            3, 1.2686
            3.25, 0.7147
            3.5, 0.9237
            3.75, 0.7297
            4, 0.7105
            4.25, 0.8683
            4.5, 0.7406
            4.75, 0.7314
            5, 0.6232"""
        data = pd.read_csv(StringIO(data), header=None, index_col=[0])
        y = data.values
        X = data.index.values
        vasicek = Vasicek()
        vasicek.fit(X, y)
        self.assertEqual(vasicek.k, 3.1286885821300157)
        self.assertEqual(vasicek.theta, 0.9074884514128012)
        self.assertEqual(vasicek.sigma, 0.553151436210664)

    # pylint: disable=too-many-function-args
    def test_historical_fit(self):
        """Test MLE fit on historical UST data from pyesg.datasets"""
        data = load_ust_historical()
        X = pd.to_datetime(
            data.index.get_level_values("year").astype(str)
            + "-"
            + data.index.get_level_values("month").astype(str)
        )
        X = ((X - X[0]) / np.timedelta64(365, "D")).values
        y = data.iloc[:, 0].values

        model = Vasicek()
        model.fit(X, y)
        self.assertTrue(all(x > 0 for x in model._fitted_params.values()))

    def test_sample(self):
        """Test sampling from a vasicek process with fixed params"""
        model = Vasicek()
        model.k, model.theta, model.sigma = 0.15, 0.045, 0.015
        samples = model.sample(1000, 30, 12, 0.03)
        self.assertEqual(samples.shape, (1000, 30 * 12 + 1))
