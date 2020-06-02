"""Ornstein-Uhlenbeck Process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Generalized Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW

    Examples
    --------
    >>> oup = OrnsteinUhlenbeckProcess.example()
    >>> oup
    <pyesg.OrnsteinUhlenbeckProcess(mu=0.05, sigma=0.015, theta=0.15)>

    >>> oup.drift(x0=0.05)
    array([0.])

    >>> oup.diffusion(x0=0.03)
    array([0.015])

    >>> oup.expectation(x0=0.03, dt=0.5)
    array([0.0315])

    >>> oup.standard_deviation(x0=0.03, dt=0.5)
    array([0.0106066])

    >>> oup.step(x0=0.03, dt=1.0, random_state=42)
    array([0.04045071])
    """

    def __init__(self, mu: float, sigma: float, theta: float) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma, theta=self.theta)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        return x0 + dx

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return self.theta * (self.mu - x0)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion of an Ornstein-Uhlenbeck process does not depend on x0, but
        # we want to match the shape of the passed x0 array
        return np.full_like(x0, self.sigma, dtype=np.float64)

    @classmethod
    def example(cls):
        return cls(mu=0.05, sigma=0.015, theta=0.15)
