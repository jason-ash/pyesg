"""Ho-Lee Process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


class HoLeeProcess(StochasticProcess):
    """
    Ho-Lee process: dX = θdt + σdW

    Examples
    --------
    >>> hlp = HoLee.example()
    >>> hlp
    <pyesg.HoLeeProcess(sigma=0.02, theta=0.005)>

    >>> hlp.drift(x0=0.045)
    array([0.005])

    >>> hlp.diffusion(x0=0.045)
    array([0.02])

    >>> hlp.expectation(x0=0.03, dt=0.5)
    array([0.0325])

    >>> hlp.standard_deviation(x0=0.03, dt=0.5)
    array([0.01])

    >>> hlp.step(x0=0.03, dt=1.0, random_state=42)
    array([0.03372067])

    >>> hlp.step(x0=[0.03], dt=1.0, random_state=42)
    array([0.03372067])
    """

    def __init__(self, sigma: float, theta: float) -> None:
        super().__init__()
        self.sigma = sigma
        self.theta = theta

    def coefs(self) -> Dict[str, float]:
        return dict(sigma=self.sigma, theta=self.theta)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        return x0 + dx

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return np.full_like(x0, self.theta, dtype=np.float64)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return np.full_like(x0, self.sigma, dtype=np.float64)

    @classmethod
    def example(cls):
        return cls(sigma=0.02, theta=0.005)
