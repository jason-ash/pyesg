"""Unit tests for pyesg datasets"""
import unittest
import pandas as pd

from pyesg.datasets import load_us_stocks, load_ust_historical


class BaseDatasetMixin:  # pylint: disable=too-few-public-methods
    """Holds common tests for all datasets"""

    def test_dataset_not_empty(self):
        """Make sure there's actually data in the dataset we're loading"""
        self.assertGreater(self.dataset.shape[0], 0)
        self.assertGreater(self.dataset.shape[1], 0)


class TestUSStocksDataset(BaseDatasetMixin, unittest.TestCase):
    """Test the US Stocks Dataset"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_us_stocks()

    def test_timespan(self):
        """Make sure the data covers the expected timespan"""
        # this data starts on 12/12/1980 and ends on 10/31/2019
        start = self.dataset.index[0]
        end = self.dataset.index[-1]
        self.assertEqual(start, pd.Timestamp("1980-12-12 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2019-10-31 00:00:00"))


class TestUSTreasuryDataset(BaseDatasetMixin, unittest.TestCase):
    """Test the US Treasury Dataset"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_ust_historical()

    def test_timespan(self):
        """Make sure the data covers the expected timespan"""
        # this data starts in April 1953 and currently ends in December 2019
        start = self.dataset.index[0]
        end = self.dataset.index[-1]
        self.assertEqual(start, (1953, 4))
        self.assertEqual(end, (2019, 12))
