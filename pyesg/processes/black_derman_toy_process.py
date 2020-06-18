"""Black-Derman-Toy Process"""
from typing import Dict
import numpy as np

from pyesg.processes.ho_lee_process import HoLeeProcess


class BlackDermanToyProcess(HoLeeProcess):
    """
    Black-Derman-Toy process: dln(X) = θdt + σdW

    Examples
    --------
    >>> bdt = BlackDermanToyProcess.example()
    >>> bdt
    <pyesg.BlackDermanToyProcess(sigma=0.015, theta=0.005)>

    >>> bdt.drift(x0=0.045)
    array([0.005])

    >>> bdt.diffusion(x0=0.045)
    array([0.015])

    >>> bdt.expectation(x0=0.03, dt=0.5)
    array([0.03007509])

    >>> bdt.standard_deviation(x0=0.03, dt=0.5)
    array([0.0106066])

    >>> bdt.step(x0=0.03, dt=1.0, random_state=42)
    array([0.03037586])

    >>> bdt.step(x0=[0.03, 0.03], dt=1.0, random_state=42)
    array([0.03037586, 0.03008791])
    """

    def __init__(self, sigma: float, theta: float) -> None:
        super().__init__(sigma=sigma, theta=theta)

    def coefs(self) -> Dict[str, float]:
        return dict(sigma=self.sigma, theta=self.theta)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return x0 * np.exp(dx)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return super()._drift(x0=np.log(x0))

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return super()._diffusion(x0=np.log(x0))

    @classmethod
    def example(cls):
        return cls(sigma=0.015, theta=0.005)
