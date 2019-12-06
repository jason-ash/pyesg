"""Geometric Brownian Motion"""
from typing import Dict
import numpy as np

from pyesg.processes import StochasticProcess
from pyesg.utils import to_array, Array


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion process: dX = μXdt + σXdW

    Examples
    --------
    >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    >>> gbm
    <pyesg.GeometricBrownianMotion{'mu': array([0.05]), 'sigma': array([0.2])}>
    >>> gbm.drift(x0=2.0)
    array([0.1])
    >>> gbm.diffusion(x0=2.0)
    array([0.4])
    >>> gbm.expectation(x0=1.0, dt=0.5)
    array([1.025])
    >>> gbm.standard_deviation(x0=1.0, dt=0.5)
    array([0.14142136])
    >>> gbm.step(x0=1.0, dt=1.0, random_state=42)
    array([1.14934283])
    >>> gbm.step(x0=[1.0, 15.0, 50.0], dt=1.0, random_state=42)
    array([ 1.14934283, 15.3352071 , 58.97688538])
    >>> gbm.logpdf(x0=1.0, xt=1.1, dt=1.0)
    array([0.65924938])
    """

    def __init__(self, mu: Array, sigma: Array) -> None:
        super().__init__()
        self.mu = to_array(mu)
        self.sigma = to_array(sigma)

    def coefs(self) -> Dict[str, np.ndarray]:
        return dict(mu=self.mu, sigma=self.sigma)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return self.mu * x0

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return self.sigma * x0


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
