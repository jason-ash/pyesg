"""Black-Karasinski Process"""
from typing import Dict
import numpy as np

from pyesg.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess


class BlackKarasinskiProcess(OrnsteinUhlenbeckProcess):
    """
    Black-Karasinski process: dln(X) = θ(μ - ln(X)) + σdW

    Examples
    --------
    >>> bkp = BlackKarasinskiProcess.example()
    >>> bkp
    <pyesg.BlackKarasinskiProcess(mu=0.01, sigma=0.005, theta=0.01)>

    >>> bkp.drift(x0=0.045)
    array([0.03111093])

    >>> bkp.diffusion(x0=0.03)
    array([0.03516558])

    >>> bkp.expectation(x0=0.03, dt=0.5)
    array([0.03053215])

    >>> bkp.standard_deviation(x0=0.03, dt=0.5)
    array([0.02486582])

    >>> bkp.step(x0=0.03, dt=1.0, random_state=42)
    array([0.03162128])

    >>> bkp.step(x0=[0.03, 0.03], dt=1.0, random_state=42)
    array([0.03162128, 0.03092302])
    """

    def __init__(self, mu: float, sigma: float, theta: float) -> None:
        super().__init__(mu=mu, sigma=sigma, theta=theta)

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma, theta=self.theta)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return x0 * np.exp(dx)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return super()._drift(x0=np.log(x0))

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return super()._drift(x0=np.log(x0))

    @classmethod
    def example(cls):
        return cls(mu=0.01, sigma=0.005, theta=0.01)
