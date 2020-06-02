"""Cox-Ingersoll-Ross Process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


class CoxIngersollRossProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross process: dX = θ(μ - X)dt + σX**0.5dW

    Examples
    --------
    >>> cir = CoxIngersollRossProcess.example()
    >>> cir
    <pyesg.CoxIngersollRossProcess(mu=0.05, sigma=0.02, theta=0.1)>

    >>> cir.drift(x0=0.045)
    array([0.0005])

    >>> cir.diffusion(x0=0.03)
    array([0.0034641])

    >>> cir.expectation(x0=0.03, dt=0.5)
    array([0.031])

    >>> cir.standard_deviation(x0=0.03, dt=0.5)
    array([0.00244949])

    >>> cir.step(x0=0.03, dt=1.0, random_state=42)
    array([0.03372067])

    >>> cir.step(x0=[0.03], dt=1.0, random_state=42)
    array([0.03372067])
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
        return self.sigma * x0 ** 0.5

    @classmethod
    def example(cls):
        return cls(mu=0.05, sigma=0.02, theta=0.1)
