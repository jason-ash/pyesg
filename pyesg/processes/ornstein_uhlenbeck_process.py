"""Ornstein-Uhlenbeck Process"""
from typing import Dict
import numpy as np

from pyesg.processes import StochasticProcess
from pyesg.utils import to_array, Array


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Generalized Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW

    Examples
    --------
    >>> ou = OrnsteinUhlenbeckProcess(mu=0.05, sigma=0.015, theta=0.15)
    >>> ou.drift(x0=0.05)
    array([0.])
    >>> ou.diffusion(x0=0.03)
    array([0.015])
    >>> ou.expectation(x0=0.03, dt=0.5)
    array([0.0315])
    >>> ou.standard_deviation(x0=0.03, dt=0.5)
    array([0.0106066])
    >>> ou.step(x0=0.03, dt=1.0, random_state=42)
    array([0.04045071])
    >>> ou.logpdf(x0=0.05, xt=0.09, dt=1.0)
    array([-0.27478901])
    """

    def __init__(self, mu: Array, sigma: Array, theta: Array) -> None:
        super().__init__()
        self.mu = to_array(mu)
        self.sigma = to_array(sigma)
        self.theta = to_array(theta)

    def coefs(self) -> Dict[str, np.ndarray]:
        return dict(mu=self.mu, sigma=self.sigma, theta=self.theta)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return self.theta * (self.mu - x0)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion of an Ornstein-Uhlenbeck process does not depend on x0
        return self.sigma


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
