"""Vasicek stochastic short-rate model"""
from typing import Dict, Optional, Union
import numpy as np

from scipy.optimize import minimize

from pyesg.models import DiffusionProcess


class Vasicek(DiffusionProcess):
    """
    Vasicek stochastic short-rate model

    r(t+dt) = r(t) + k*(theta - r(t))*dt + sigma*dt**0.5*norm(0,1)

    Parameters
    ----------
    k : float, the rate of mean reversion - fit by MLE
    theta : float, the long-term interest rate level - fit by MLE
    sigma : float, the volatility of rates - fit by MLE
    """

    def __init__(self) -> None:
        self.k: Optional[float] = None  # fit parameter
        self.theta: Optional[float] = None  # fit parameter
        self.sigma: Optional[float] = None  # fit parameter

    def __call__(
        self, x: float, dt: float, dW: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        """Discrete approximation of the continuous Vasicek diffusion process"""
        if dW is None:
            dW = np.random.randn()
        if self._check_fitted():
            return self.k * (self.theta - x) * dt + self.sigma * dt ** 0.5 * dW
        raise RuntimeError("Must call 'fit' first!")

    @classmethod
    def _log_likelihood(
        cls, X: np.ndarray, y: np.ndarray, k: float, theta: float, sigma: float
    ) -> float:
        """
        Log likelihood function, given model parameters and observed data

        Reference
        ---------
        https://pdfs.semanticscholar.org/dc0c/75b5d0002607046277a6768b9b54f34877c3.pdf
        """
        B = (1 - np.exp(-k * (X[1:] - X[:-1]))) / k
        m = theta * k * B + y[:-1] * (1 - k * B)
        v = sigma ** 2 * (B - k / 2 * B ** 2)
        return (np.log(2 * np.pi * v) + (y[1:] - m) ** 2 / v).sum() / -2

    @property
    def _fitted_params(self) -> Dict:
        """
        Returns a dictionary of fitted model parameters.
        Parameters default to None if they haven't been fitted yet.
        """
        return dict(k=self.k, theta=self.theta, sigma=self.sigma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Vasicek":
        """
        Fits the parameters of the Vasicek model using maximum likelihood.

        Parameters
        ----------
        X : np.array of timesteps in years; must be increasing
            e.g. array([ 0.08333333,  0.16666667,  0.25]
        y : np.array of short-rates at each timestep
            e.g. array([2.190e-02, 2.160e-02, 2.110e-02]
        """
        # minimize the negative log-likelihood
        f = lambda params, X, y: -1 * self._log_likelihood(X, y, *params)
        mle = minimize(f, x0=[0.5, 0.1, 0.1], args=(X, y), method="Nelder-Mead")
        if mle.success:
            self.k, self.theta, self.sigma = mle.x
            return self
        raise RuntimeError("Model failed to converge")
