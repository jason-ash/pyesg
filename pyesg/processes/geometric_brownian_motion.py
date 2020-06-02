"""Geometric Brownian Motion / Black Merton Scholes Process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion: dX = X*exp((μ - δ - (1/2)*σ**2)dt + σdW)

    Examples
    --------
    >>> gbm = GeometricBrownianMotion.example()
    >>> gbm
    <pyesg.GeometricBrownianMotion(mu=0.05, sigma=0.2, dividend=0.01)>

    >>> gbm.drift(10.0)
    array([0.02])

    >>> gbm.diffusion(10.0)
    array([0.2])

    >>> gbm.expectation(10.0, 0.5)
    array([10.10050167])

    >>> gbm.standard_deviation(10.0, 0.5)
    array([0.14142136])

    >>> gbm.step(10.0, dt=0.5, random_state=42)
    array([10.83553577])
    """

    def __init__(self, mu: float, sigma: float, dividend: float = 0.0) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.dividend = dividend

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma, dividend=self.dividend)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return x0 * np.exp(dx)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        drift = self.mu - self.dividend - 0.5 * self.sigma * self.sigma
        return np.full_like(x0, drift, dtype=np.float64)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return np.full_like(x0, self.sigma, dtype=np.float64)

    @classmethod
    def example(cls):
        return cls(mu=0.05, sigma=0.2, dividend=0.01)
