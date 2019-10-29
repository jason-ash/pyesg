"""Estimator classes for interest rate curve interpolation"""
from typing import Optional, Union

import numpy as np

from scipy.optimize import least_squares


class Interpolator:
    """Base class for Interpolators"""


class NelsonSiegel(Interpolator):
    """
    Nelson-Siegel Curve Interpolator

    Parameters
    ----------
    tau : float, optional, if provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process.
    """

    def __init__(self, tau: Optional[float] = None) -> None:
        self._fit_tau = tau is None
        self.tau = tau

    def __repr__(self):
        return "Nelson-Siegel Interpolator"

    @classmethod
    def formula(
        cls,
        t: Union[float, np.ndarray],
        beta0: float,
        beta1: float,
        beta2: float,
        tau: float,
    ) -> Union[float, np.ndarray]:
        """Returns the Nelson-Siegel estimate of a rate at maturity t"""
        return (
            beta0
            + beta1 * (1 - np.exp(-t * tau)) / (t * tau)
            + beta2 * ((1 - np.exp(-t * tau)) / (t * tau) - np.exp(-t * tau))
        )

    @property
    def _check_fitted(self) -> bool:
        """Returns a boolean indicating whether or not the model has been fitted"""
        attributes = ["beta0", "beta1", "beta2", "tau"]
        return all([hasattr(self, a) for a in attributes])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NelsonSiegel":
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
            def f(params, t, y):
                return self.formula(t, *params) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 1.0], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.tau = ls.x
        else:
            # keep tau fixed; solve for all betas
            def f(params, t, y):
                assert self.tau is not None
                return self.formula(t, *params, tau=self.tau) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2 = ls.x
        return self

    def predict(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self._check_fitted:
            assert self.tau is not None
            return self.formula(X, self.beta0, self.beta1, self.beta2, self.tau)
        else:
            raise RuntimeError("Must call 'fit' first!")


class NelsonSiegelSvensson(Interpolator):
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
        self._fit_tau = (tau0 is None) or (tau1 is None)
        self.tau0 = tau0
        self.tau1 = tau1

    def __repr__(self):
        return "Nelson-Siegel-Svensson Interpolator"

    @classmethod
    def formula(
        cls,
        t: Union[float, np.ndarray],
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        tau0: float,
        tau1: float,
    ) -> Union[float, np.ndarray]:
        """Returns the Nelson-Siegel-Svensson estimate of a rate at maturity t"""
        return (
            beta0
            + beta1 * (1 - np.exp(-t * tau0)) / (t * tau0)
            + beta2 * ((1 - np.exp(-t * tau0)) / (t * tau0) - np.exp(-t * tau0))
            + beta3 * ((1 - np.exp(-t * tau1)) / (t * tau1) - np.exp(-t * tau1))
        )

    @property
    def _check_fitted(self) -> bool:
        attributes = ["beta0", "beta1", "beta2", "beta3", "tau0", "tau1"]
        return all([hasattr(self, a) for a in attributes])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NelsonSiegelSvensson":
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
            def f(params, t, y):
                return self.formula(t, *params) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 0.01, 1.0, 1.0], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau0, self.tau1 = ls.x
        else:
            # keep taus fixed; solve for all betas
            def f(params, t, y):
                assert self.tau0 is not None
                assert self.tau1 is not None
                return self.formula(t, *params, tau0=self.tau0, tau1=self.tau1) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.beta3 = ls.x
        return self

    def predict(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self._check_fitted:
            assert self.tau0 is not None
            assert self.tau1 is not None
            return self.formula(
                X, self.beta0, self.beta1, self.beta2, self.beta3, self.tau0, self.tau1
            )
        else:
            raise RuntimeError("Must call 'fit' first!")
