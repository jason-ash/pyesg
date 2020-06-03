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
from pyesg.utils import RandomState


def interpolate(scenarios: np.ndarray, maturities: Optional[list] = None) -> np.ndarray:
    """
    This model uses a simplified Nelson-Siegel interpolation formula to translate
    a "short rate" with a duration of 1 year and a "long rate" wtih a duration of
    20 years into a full yield curve with any number of maturities.Normally, the
    Nelson-Siegel formula uses β_0, β_1, β_2, and 𝜏 as free variables, but this
    model sets 𝜏=0.4 and β_2=0, leaving just β_0 and β_1 as free variables. Because
    we have two rates and two free variables, we solve a system of equations for
    the values of β_0 and β_1.
    """

    def factor(t: float, tau: float = 0.4) -> float:
        # Nelson-Siegel factor, which varies by maturity and tau
        return (1 - np.exp(-tau * t)) / (tau * t)

    def beta(scenarios: np.ndarray) -> np.ndarray:
        # Returns an array of (n_scenarios, n_steps, 2); columns for β_0 and β_1
        r20 = scenarios[:, :, 0]
        r1 = scenarios[:, :, 0] - scenarios[:, :, 1]

        betas = np.zeros(shape=scenarios[:, :, :2].shape)
        betas[:, :, 1] = (r20 - r1) / (factor(20) - factor(1))
        betas[:, :, 0] = r20 - betas[:, :, 1] * factor(20)
        return betas

    # what yield curve maturities do we want to end up with? The Academy uses these
    # 10 values, which will be used, unless overwritten by the user
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30] or maturities
    mat_array = factor(np.array(maturities))
    mat_array = np.vstack([np.ones_like(mat_array), mat_array])
    return beta(scenarios) @ mat_array


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
    >>> model = AcademyRateModel(long_rate=0.0225, spread=0.0066, volatility=0.0287)
    >>> model
    <pyesg.AcademyRateModel(long_rate=0.0225, spread=0.0066, volatility=0.0287)>
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
        coefs = dict(
            long_rate=self.long_rate, spread=self.spread, volatility=self.volatility
        )
        params = (f"{k}={repr(v)}" for k, v in coefs.items())
        return f"<pyesg.{self.__class__.__qualname__}({', '.join(params)})>"

    def perturb(self, scenarios: np.ndarray, floor: float = 0.0001) -> np.ndarray:
        """
        Handles the perturbing function from the model for the first 12 months of the
        projection, where scenarios have already been interpolated.
        """
        perturbation_coefficient = np.zeros(shape=scenarios[:, :, :1].shape)
        vals = np.arange(12, 0, -1) / 12
        perturbation_coefficient[:, :12, :] = vals[None, :, None]

        perturbation_value = scenarios[:, 0, :] - self.yield_curve.values

        perturbation = perturbation_coefficient * perturbation_value[:, None, :]
        return np.maximum(floor, scenarios - perturbation)

    def _scenarios(
        self, dt: float, n_scenarios: int, n_steps: int, random_state: RandomState
    ) -> np.ndarray:
        """
        Returns the raw scenarios from the AcademyRateProcess, without any interpolation
        or perturbation. Intentionally "hidden" with the leading underscore because the
        primary scenarios should have interpolation and perturbation by default.
        """
        return self.process.scenarios(
            x0=[self.long_rate, self.spread, self.volatility],
            dt=dt,
            n_scenarios=n_scenarios,
            n_steps=n_steps,
            random_state=random_state,
        )

    def scenarios(
        self, n_scenarios: int, n_steps: int, random_state: RandomState
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
        # can only handle dt = monthly right now
        dt = 1 / 12

        scenarios = self._scenarios(
            dt=dt, n_scenarios=n_scenarios, n_steps=n_steps, random_state=random_state
        )
        interpolated_scenarios = interpolate(scenarios)
        return self.perturb(interpolated_scenarios)
