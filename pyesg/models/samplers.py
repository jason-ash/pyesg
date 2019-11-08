"""Algorithms for sampling from stochastic processes"""
from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy.stats import norm


def recursive_sampler(
    process: Callable,
    params: Dict[str, float],
    n_scen: int,
    n_years: int,
    step_size: int,
    init: Union[float, np.ndarray],
    random_state: Optional[int] = None,
):
    """
    Returns an array of samples from a recursively-defined process.

    Parameters
    ----------
    process : Callable, a function that calculates value(t+dt) from value(t) and a set
        of parameters. Must have arguments arranged by parameters first, then the start
        value, e.g. value(t)
    params : Dictionary[str, float], the required parameters to evaluate the process, as
        "parameter_name": parameter_value; will be passed as **params to process.
    n_scen : int, the number of scenarios to generate
    n_years : int, the number of years per scenario
    step_size : int, the number of steps per year; e.g. 1 for annual time steps, 12 for
        monthly, 24 for bi-weekly, 52 for weekly, 252 (or 365) for daily
    init : Union[float, np.ndarray], either a single start value that will be broadcast
        to all scenarios, or a start value array that should match the shape of "n_scen"
    random_state : Optional[int], to ensure repeated results if desired. If None, then
        results will be created with no random seed

    Returns
    -------
    scenarios : np.ndarray with shape (n_scen, 1 + n_years*step_size), with the scenario
        results from the process
    """
    # create an array of random numbers we'll need to generate the scenarios
    # then overwrite the first value of each scenario equal as "init"
    # this is currently assumed to be a normal distribution - come back
    # later to potentially add some flexibility to use other distributions
    out = norm.rvs(size=(n_scen, 1 + n_years * step_size), random_state=random_state)
    out[:, 0] = init

    # recursive calls operate on rows, but we can parallelize over scenarios
    for i in range(n_years * step_size):
        out[:, i + 1] = process(**params, init=out[:, i], dW=out[:, i + 1])
    return out
