"""
Implements the American Academy of Actuaries stochastic interest rate model, including
stochastic rate generation, yield curve interpolation, flooring, scenario significance
testing, and yield curve perturbations.

References
----------
https://www.actuary.org/sites/default/files/pdf/life/lbrc_dec08.pdf, page 8
https://www.soa.org/globalassets/assets/files/research/research-ecs-release-notes-v7.pdf
"""
from typing import Optional
import numpy as np
import pandas as pd

from pyesg.processes.academy_rate_process import AcademyRateProcess
from pyesg.utils import Array, RandomState

# pylint: disable=too-many-arguments


def interpolate(
    short_rate: np.ndarray,
    long_rate: np.ndarray,
    short_maturity: float = 1.0,
    long_maturity: float = 20.0,
    interpolated_maturities: Optional[Array] = None,
    tau: float = 0.4,
) -> np.ndarray:
    """
    The American Academy of Actuaries uses a simplified Nelson-Siegel with two free
    parameters, β_0 and β_1. β_2 is set to zero and 𝜏 is set to 0.4. It is typically
    calibrated with a 1-year rate and a 20-year rate, which allows us to solve a system
    of equations for β_0 and β_1. A rate at maturity t, r(t) is equal to:

        r(t) = β_0 + β_1 * (1 - exp(-𝜏 * t)) / (𝜏 * t)

    Parameters
    ----------
    short_rate : np.ndarray, an array with shape (n_scenarios, n_steps) that contains
        the short rate values
    long_rate : np.ndarray, an array with shape (n_scenarios, n_steps) that contains
        the long rate values
    short_rate_maturity : float, default 1.0, the duration of the short rate
    long_rate_maturity : float, default 20.0, the duration of the short rate
    interpolated_maturities : Array of floats or ints, determines the interpolated
        maturities to export. By default, uses [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    tau : float, default 0.4, the parameter 𝜏 in the Nelson-Siegel interpolator

    Returns
    -------
    rates : np.ndarray, an array of rates with shape (n_scenarios, n_steps, maturities)
    """
    # the nelson-siegel factor in the equation r(t) = b_0 + b_1 * factor(t)
    f = lambda t: (1 - np.exp(-tau * t)) / (tau * t)

    # an array of the betas that fit each time step of rates provided
    betas = np.zeros(shape=(short_rate.shape[0], short_rate.shape[1], 2))
    betas[:, :, 1] = (long_rate - short_rate) / (f(long_maturity) - f(short_maturity))
    betas[:, :, 0] = long_rate - betas[:, :, 1] * f(long_maturity)

    # finally, we use the nelson-sigel formula to calculate rates at each maturity
    # if interpolated maturities weren't explicitly provided, then we use the defaults
    if interpolated_maturities is None:
        # if this value wasn't provided, use the standard defaults from the AAA
        interpolated_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

    maturities = f(np.array(interpolated_maturities))
    maturities = np.vstack([np.ones_like(maturities), maturities])
    return betas @ maturities


def perturb(
    scenarios: np.ndarray, n_steps: int, yield_curve: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    The Academy model "perturbs" scenarios in the first several periods so that they
    match more closely to the starting yield curve today. The default model uses monthly
    time steps, and perturbs values according to a decreasing factor for each month in
    the first projection year.

    Parameters
    ----------
    scenarios : np.ndarray, a rates array with shape (n_scenarios, n_steps, maturities)
    n_steps : int, the number of time steps over which the perturbations should occur,
        e.g. 12 in a monthly projection to grade rates over the first projection year.
    yield_curve : np.ndarray, a yield curve array with shape (maturities) representing
        the current interest rate curve.

    Returns
    -------
    rates : np.ndarray, an array of rates with shape (n_scenarios, n_steps, maturities)
    """
    factor_array = np.zeros(shape=(scenarios.shape[0], scenarios.shape[1], 1))
    factor = np.arange(n_steps, 0, -1) / n_steps
    factor_array[:, :n_steps, :] = factor[None, :, None]

    difference = scenarios[:, 0, :] - yield_curve

    perturbation = factor_array * difference[:, None, :]
    return scenarios - perturbation


def scenario_significance_value(scenarios: np.ndarray) -> np.ndarray:
    """
    Returns the scenario significance value for an array of scenarios with shape
    (n_scenarios, n_steps, maturities). Returns an array with shape (n_scenarios,) where
    each value in the array is the significance value of each scenario.

    Parameters
    ----------
    scenarios : np.ndarray, a rates array with shape (n_scenarios, n_steps, maturities)

    Returns
    -------
    significance : np.ndarray, an array of significance values per scenario, with shape
        (n_scenarios,)
    """
    # this is an array of discount factors based on the 20-year rate of the scenarios
    # for now assume the 20 year rate is the second-to-last column in the scenario array
    # TODO : update this to dynamically handle which column to use
    discount_factor = (1 + scenarios[:, :, -2] / 2) ** (-1 / 3)
    discount_factor[:, 0] = 1.0
    discount_factor = discount_factor.cumprod(axis=1).cumsum(axis=1) - 1.0
    return discount_factor[:, -1] ** 0.5


def scenario_rank(scenarios: np.ndarray) -> np.ndarray:
    """
    Returns an array with the rank ordering of a batch of scenarios. The scenarios must
    have shape (n_scenarios, n_steps, maturities), and the returned array will have
    shape (n_scenarios,). The scenarios are ranked based on significance values, ordered
    from highest (rank 1 / N) to lowest (rank N / N).

    Parameters
    ----------
    scenarios : np.ndarray, a rates array with shape (n_scenarios, n_steps, maturities)

    Returns
    -------
    rank : np.ndarray, an array with each scenario's rank, with shape (n_scenarios,)
    """
    # we just rank scenarios from highest value to lowest value based on significance
    # np.argsort ranks lowest to highest by default, so we negate the original arrray
    significance = scenario_significance_value(scenarios)
    return (-1 * significance).argsort()


def scenario_subset(scenarios: np.ndarray, size: int) -> np.ndarray:
    """
    Returns a subset of scenarios from a scenario array. This was originally intended to
    take a set of 10,000 scenarios and return subsets of 50, 200, 500, and 1000, but can
    be more flexible, provided the size parameter evenly divides the number of scenarios
    in the file.

    The algorithm ranks each scenario based on its significance value from highest to
    lowest. Then it takes every Nth scenario in order to retrieve the desired subset.

    For example, if you start with 10,000 scenarios and want to pick a representative
    sample of 50, we take ranked scenario 100, 300, 500, ..., 9900. For 200 scenarios we
    take ranked scenario 25, 75, 125, 175, 225, ..., 9975.

    Parameters
    ----------
    scenarios : np.ndarray, a rates array with shape (n_scenarios, n_steps, maturities)

    Returns
    -------
    subset : np.ndarray, a subset of the original scenarios array with a new shape of
        (size, n_steps, maturities)
    """
    # we are indexing the "ranks" array, choosing evenly spaced scenarios based on rank
    # if we find that the resulting array doesn't meet the size, then we raise an error
    n_scenarios = scenarios.shape[0]
    if n_scenarios < size:
        raise RuntimeError(
            f"Incompatible subset request: {size} scenarios from {n_scenarios} total."
        )
    start, step = n_scenarios // size // 2, n_scenarios // size
    idx = np.arange(start, n_scenarios, step)
    if len(idx) != size:
        raise RuntimeError(
            f"Incompatible subset request: {size} scenarios from {n_scenarios} total."
        )
    ranks = scenario_rank(scenarios)
    return scenarios[ranks[idx], :, :]


class AcademyRateModel:
    """
    This class implements the American Academy of Actuaries stochastic rate model.

    This model combines several parts:
        1. Generating interest rates from a stochastic volatility process
        2. Interpolating a yield curve from the short- and long-rates from (1)
        3. Flooring interest rates
        4. Perturbing rates within the first year of the projection

    It also has functionality to calculate the "significance value" of a scenario, which
    may be used to filter scenarios for different characteristics.

    Examples
    --------
    >>> model = AcademyRateModel()
    >>> model
    <pyesg.AcademyRateModel>
    """

    def __init__(self, volatility: float = 0.0287) -> None:
        self.volatility = volatility
        self.process = AcademyRateProcess(
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
        self.yield_curve = pd.Series(
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
        self.long_rate = self.yield_curve.loc[20]
        self.spread = self.yield_curve.loc[20] - self.yield_curve.loc[1]

    def __repr__(self) -> str:
        """Returns a string representation of this model"""
        return f"<pyesg.{self.__class__.__qualname__}>"

    def scenarios(
        self,
        dt: float,
        n_scenarios: int,
        n_steps: int,
        floor: float = 0.0001,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """
        Generate scenarios using this model. The returned scenarios will be interpolated
        along a full yield curve using a simplified Nelson-Siegel model, and values in
        the first year will be "perturbed" according to their difference from the actual
        yield curve observed today.

        Parameters
        ----------
        dt : float, the length between steps, in years, e.g. 1/12 for monthly steps
        n_scenarios : int, the number of scenarios to generate, e.g. 1000
        n_steps : int, the number of steps in the scenario, e.g. 52. In combination with
            dt, this determines the scope of the scenario, e.g. dt=1/12 and n_step=360
            will produce 360 monthly time steps, i.e. a 30-year monthly projection
        random_state : Union[int, np.random.RandomState, None], either an integer seed
            or a numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        scenarios : np.ndarray, will have shape (n_scenarios, n_steps + 1, 10), where 10
            represents 10 points on the yield curve for each scenario.
        """
        # create the raw scenarios from the underlying AcademyRateProcess, which outputs
        # an array with shape: (n_scenarios, n_steps, [long_rate, spread, volatility])
        scenarios = self.process.scenarios(
            x0=[self.long_rate, self.spread, self.volatility],
            dt=dt,
            n_scenarios=n_scenarios,
            n_steps=n_steps,
            random_state=random_state,
        )

        # interpolate the short and long rates into a full yield curve across a default
        # range of maturities from 0.25 years to 30 years.
        interpolated_scenarios = interpolate(
            short_rate=scenarios[:, :, 0] - scenarios[:, :, 1],
            long_rate=scenarios[:, :, 0],
        )

        # "perturb" the scenarios by reducing the difference between the starting yield
        # curve and the generated scenarios during the first year.
        perturbed_scenarios = perturb(
            scenarios=interpolated_scenarios,
            n_steps=int(1 / dt),
            yield_curve=self.yield_curve.values,
        )

        # floor the scenarios before returning; default is 0.0001
        return np.maximum(floor, perturbed_scenarios)
