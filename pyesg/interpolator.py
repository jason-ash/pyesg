"""Estimator classes for interest rate curve interpolation"""
from typing import Dict, Optional, Union
import numpy as np
from scipy import optimize


class Interpolator:
    """Base class for Interpolators"""

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}>"

    def __call__(
        self, x: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        """Returns the Interpolator estimate of a rate at maturity x"""
        raise NotImplementedError()

    @property
    def _fitted_params(self) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of fitted model parameters.
        Parameters should default to None if they haven't been fitted yet.
        """
        raise NotImplementedError()

    def _check_fitted(self) -> bool:
        """Returns a boolean indicating whether or not the model has been fitted"""
        return all(x is not None for x in self._fitted_params.values())

    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self._check_fitted():
            return self(x, **self._fitted_params)
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
        self.tau0 = tau0  # optionally fit parameter
        self.tau1 = tau1  # optionally fit parameter
        self._fit_tau = (tau0 is None) or (tau1 is None)
        self.beta0: Optional[float] = None  # fit parameter
        self.beta1: Optional[float] = None  # fit parameter
        self.beta2: Optional[float] = None  # fit parameter
        self.beta3: Optional[float] = None  # fit parameter

    def __call__(
        self, x: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        beta0 = params["beta0"]
        beta1 = params["beta1"]
        beta2 = params["beta2"]
        beta3 = params["beta3"]
        tau0 = params["tau0"]
        tau1 = params["tau1"]
        factor0 = (1 - np.exp(-x * tau0)) / (x * tau0)
        factor1 = (1 - np.exp(-x * tau1)) / (x * tau1)
        return (
            beta0
            + beta1 * factor0
            + beta2 * (factor0 - np.exp(-x * tau0))
            + beta3 * (factor1 - np.exp(-x * tau1))
        )

    @property
    def _fitted_params(self) -> Dict:
        return dict(
            beta0=self.beta0,
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            tau0=self.tau0,
            tau1=self.tau1,
        )

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
