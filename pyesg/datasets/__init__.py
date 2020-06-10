"""Utility functions for loading useful datasets"""
import pathlib
from typing import Any, Dict, Union
import numpy as np
import pandas as pd


def _read_data(file_name: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Reads a data file from pyesg/data. Optionally provide kwargs to pd.read_csv"""
    directory = pathlib.Path(__file__).parents[0]
    file_path = pathlib.Path(directory, file_name).resolve()
    return pd.read_csv(file_path, **kwargs)


def load_academy_sample_scenario() -> Dict[str, Any]:
    """
    Returns a dictionary with the inputs and outputs of a single scenario from the
    American Academy of Actuaries stochastic interest rate model. Given these inputs,
    including a specific random seed, the scenario values should be able to be
    reproduced exactly, which can be useful for validating models.

    Returns
    -------
    scenario : Dict[str, Any], a dictionary with parameter inputs and outputs
    """
    output: Dict[str, Any] = {}
    output["process_parameters"] = dict(
        beta1=0.00509,
        beta2=0.02685,
        beta3=0.04001,
        rho12=-0.19197,
        rho13=0.0,
        rho23=0.0,
        sigma2=0.04148,
        sigma3=0.11489,
        tau1=0.035,
        tau2=0.01,
        tau3=0.0287,
        theta=1.0,
        phi=0.0002,
        psi=0.25164,
        long_rate_max=0.18,
        long_rate_min=0.0115,
    )
    output["yield_curve"] = pd.Series(
        index=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
        data=[
            0.0155,
            0.0160,
            0.0159,
            0.0158,
            0.0162,
            0.0169,
            0.0183,
            0.0192,
            0.0225,
            0.0239,
        ],
        name="yield_curve",
    )
    output["volatility"] = 0.0287
    output["long_rate"] = output["yield_curve"].loc[20]
    output["spread"] = output["yield_curve"].loc[20] - output["yield_curve"].loc[1]
    output["dt"] = 1 / 12
    output["n_scenarios"] = 1
    output["n_steps"] = 360
    output["floor"] = 0.0001
    output["random_state"] = 42
    output["sample_scenario"] = _read_data("academy_sample_scenario.csv", header=None)
    output["sample_scenario_significance_value"] = np.array([12.03111627])
    return output


def load_us_stocks(**kwargs) -> pd.DataFrame:
    """
    Returns a DataFrame with historical stock prices for AAPL, MSFT, and AMZN.

    Stock prices are daily close value adjusted to include total returns.

        Index : Date
        Columns : aapl, msft, amzn

    Data provided by yahoo finance: https://finance.yahoo.com
    """
    return _read_data(
        "us_stocks.csv", header=0, index_col=[0], parse_dates=[0], **kwargs
    )


def load_ust_historical(**kwargs) -> pd.DataFrame:
    """
    Returns a DataFrame of US Treasury rates for select maturities at monthly intervals
    starting from April 1953 through December 2018.

        Index : Year, Month
        Columns : 3_month, 6_month, 12_month, 24_month, 36_month, 60_month
                   84_month, 120_month, 240_month, 360_month
    """
    return _read_data("ust_historical.csv", header=0, index_col=[0, 1], **kwargs)
