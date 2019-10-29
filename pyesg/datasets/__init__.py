"""Useful Datasets"""
import pathlib
from typing import Union

import pandas as pd


def _read_data(file_name: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Reads a data file from pyesg/data. Optionally provide kwargs to pd.read_csv"""
    directory = pathlib.Path(__file__).parents[0]
    file_path = pathlib.Path(directory, file_name).resolve()
    return pd.read_csv(file_path, **kwargs)


def load_ust_historical(**kwargs) -> pd.DataFrame:
    """
    Returns a DataFrame of US Treasury rates for select maturities at monthly intervals
    starting from April 1953 through December 2018.

        Index : Year, Month
        Columns : 3_month, 6_month, 12_month, 24_month, 36_month, 60_month
                   84_month, 120_month, 240_month, 360_month
    """
    return _read_data("ust_historical.csv", header=0, index_col=[0, 1], **kwargs)
