"""Nelson-Siegel-Svensson rate curve interpolator"""
from typing import Dict, Optional, Union
import numpy as np
from scipy import optimize

from pyesg.interpolator import Interpolator


class SvenssonInterpolator(Interpolator):
    """
    Nelson-Siegel-Svensson Curve Interpolator

    Parameters
    ----------
    tau0 : float, optional, if both tau0 and tau1 are provided, then neither value is
        solved when 'fit' is called; otherwise, if neither or just one value is provided,
        then both values will be provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process
    tau1 : float, optional, if both tau0 and tau1 are provided, then neither value is
        solved when 'fit' is called; otherwise, if neither or just one value is provided,
        then both values will be provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process
    """

    def __init__(
        self, tau0: Optional[float] = None, tau1: Optional[float] = None
    ) -> None:
        self.tau0 = tau0  # optionally fit parameter
        self.tau1 = tau1  # optionally fit parameter
        self._fit_tau = (tau0 is None) or (tau1 is None)
        self.beta0: Optional[float] = None  # fit parameter
        self.beta1: Optional[float] = None  # fit parameter
        self.beta2: Optional[float] = None  # fit parameter
        self.beta3: Optional[float] = None  # fit parameter

    def __call__(
        self, X: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]
        beta3 = params["beta3"]
        tau0 = params["tau0"]
        tau1 = params["tau1"]
        factor0 = (1 - np.exp(-X * tau0)) / (X * tau0)
        factor1 = (1 - np.exp(-X * tau1)) / (X * tau1)
        return (
            beta0
            + beta1 * factor0
            + beta2 * (factor0 - np.exp(-X * tau0))
            + beta3 * (factor1 - np.exp(-X * tau1))
        )

    def coefs(self) -> Dict:
        return dict(
            beta0=self.beta0,
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            tau0=self.tau0,
            tau1=self.tau1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the Nelson-Siegel-Svensson interpolator using ordinary least squares

        Parameters
        ----------
        X : np.array of maturies, must be >0
        y : np.array of rates corresponding to each maturity

        Returns
        -------
        self : returns an instance of self
        """
        if self._fit_tau:
            # solve for all betas and taus
            def f(x0, x, y):
                return (
                    self(
                        x,
                        beta0=x0[0],
                        beta1=x0[1],
                        beta2=x0[2],
                        beta3=x0[3],
                        tau0=x0[4],
                        tau1=x0[5],
                    )
                    - y
                )

            ls = optimize.least_squares(
                f, x0=[0.1, 0.1, 0.1, 0.1, 1.0, 1.0], args=(X, y)
            )
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau0, self.tau1 = ls.x
        else:
            # keep taus fixed; solve for all betas
            def f(x0, x, y):
                return (
                    self(
                        x,
                        beta0=x0[0],
                        beta1=x0[1],
                        beta2=x0[2],
                        beta3=x0[3],
                        tau0=self.tau0,
                        tau1=self.tau1,
                    )
                    - y
                )

            ls = optimize.least_squares(f, x0=[0.01, 0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.beta3 = ls.x
        return self
