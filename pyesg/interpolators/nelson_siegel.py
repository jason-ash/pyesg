"""Nelson-Siegel rate curve interpolator"""
from typing import Dict, Optional, Union
import numpy as np
from scipy import optimize

from pyesg.interpolator import Interpolator


class NelsonSiegelInterpolator(Interpolator):
    """
    Nelson-Siegel Curve Interpolator, used to interpolate between maturities on a yield
    curve. This interpolator is defined for a given maturity, t, as:

        y(t) = (
            β_0
            + β_1 * ((1 - exp(-t * 𝜏)) / (t * 𝜏))
            + β_2 * ((1 - exp(-t * 𝜏)) / (t * 𝜏) - exp(-t * 𝜏))
        )

    Parameters
    ----------
    tau : Optional[float], if provided, this value is not solved when 'fit' is called;
        otherwise, it is considered a free variable as part of the fitting process.

    References
    ----------
    https://comisef.eu/files/wps031.pdf
    """

    def __init__(self, tau: Optional[float] = None) -> None:
        self.tau = tau  # optionally fit parameter
        self._fit_tau: bool = tau is None  # whether or not to fit tau as a parameter
        self.beta0: Optional[float] = None  # fit parameter
        self.beta1: Optional[float] = None  # fit parameter
        self.beta2: Optional[float] = None  # fit parameter

    def __call__(
        self, X: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        """Returns the Nelson-Siegel interpolated value at a point, x"""
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]
        tau = params["tau"]
        factor = (1 - np.exp(-X * tau)) / (X * tau)
        return beta0 + beta1 * factor + beta2 * (factor - np.exp(-X * tau))

    def coefs(self) -> Dict:
        return dict(beta0=self.beta0, beta1=self.beta1, beta2=self.beta2, tau=self.tau)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the Nelson-Siegel interpolator using ordinary least squares

        Parameters
        ----------
        X : np.array of maturies, must be >0
        y : np.array of rates corresponding to each maturity

        Returns
        -------
        self : returns an instance of self
        """
        if self._fit_tau:
            # solve for all betas and tau
            def f(x0, x, y):
                return self(x, beta0=x0[0], beta1=x0[1], beta2=x0[2], tau=x0[3]) - y

            ls = optimize.least_squares(f, x0=[0.01, 0.01, 0.01, 1.0], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.tau = ls.x
        else:
            # keep tau fixed; solve for all betas
            def f(x0, x, y):
                return self(x, beta0=x0[0], beta1=x0[1], beta2=x0[2], tau=self.tau) - y

            ls = optimize.least_squares(f, x0=[0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2 = ls.x
        return self
