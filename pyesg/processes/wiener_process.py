"""Wiener Process"""
from typing import Dict, List, Union
import numpy as np

from pyesg.stochastic_process import StochasticProcess
from pyesg.utils import to_array


class WienerProcess(StochasticProcess):
    """
    Generalized Wiener process: dX = μdt + σdW

    Examples
    --------
    >>> wp = WienerProcess.example()
    >>> wp
    <pyesg.WienerProcess(mu=0.05, sigma=0.2)>

    >>> wp.drift(x0=0.0)
    array([0.05])

    >>> wp.diffusion(x0=0.0)
    array([0.2])

    >>> wp.expectation(x0=0.0, dt=0.5)
    array([0.025])

    >>> wp.standard_deviation(x0=0.0, dt=0.5)
    array([0.14142136])

    >>> wp.step(x0=0.0, dt=1.0, random_state=42)
    array([0.14934283])

    >>> wp.step(x0=np.array([1.0]), dt=1.0, random_state=42)
    array([1.14934283])
    """

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        return x0 + dx

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        # drift of a Wiener process does not depend on x0
        return np.full_like(x0, self.mu, dtype=np.float64)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion of a Wiener process does not depend on x0
        return np.full_like(x0, self.sigma, dtype=np.float64)

    @classmethod
    def example(cls):
        return cls(mu=0.05, sigma=0.2)


class JointWienerProcess(StochasticProcess):
    """
    Joint Wiener processes: dX = μdt + σdW

    Examples
    --------
    >>> jwp = JointWienerProcess(
    ...     mu=[0.05, 0.03], sigma=[0.20, 0.15], correlation=[[1.0, 0.5], [0.5, 1.0]]
    ... )
    >>> jwp.drift(x0=[1.0, 1.0])
    array([0.05, 0.03])
    >>> jwp.diffusion(x0=[1.0, 1.0])
    array([[0.2       , 0.        ],
           [0.075     , 0.12990381]])
    >>> jwp.expectation(x0=[1.0, 1.0], dt=0.5)
    array([1.025, 1.015])
    >>> jwp.standard_deviation(x0=[1.0, 2.0], dt=2.0)
    array([[0.28284271, 0.        ],
           [0.10606602, 0.18371173]])
    >>> jwp.step(x0=np.array([1.0, 1.0]), dt=1.0, random_state=42)
    array([1.14934283, 1.0492925 ])
    >>> jwp.correlation = [[1.0, 0.99], [0.99, 1.0]]
    >>> jwp.step(x0=np.array([1.0, 1.0]), dt=1.0, random_state=42)
    array([1.14934283, 1.10083636])
    """

    def __init__(
        self,
        mu: Union[List[float], List[int], np.ndarray],
        sigma: Union[List[float], List[int], np.ndarray],
        correlation: Union[List[float], np.ndarray],
    ) -> None:
        super().__init__(dim=len(mu))
        self.mu = to_array(mu)
        self.sigma = to_array(sigma)
        self.correlation = to_array(correlation)

    def coefs(self) -> Dict[str, np.ndarray]:
        return dict(mu=self.mu, sigma=self.sigma, correlation=self.correlation)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        return x0 + dx

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        # mu is already an array of expected returns; it doesn't depend on x0
        return np.full_like(x0, self.mu, dtype=np.float64)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion does not depend on x0, but we want to match the shape of x0. If x0
        # has shape (100, 2), then we want to export an array with size (100, 2, 2)
        volatility = np.diag(self.sigma)
        if x0.ndim > 1:
            # we have multiple start values for each index
            volatility = np.repeat(volatility[None, :, :], x0.shape[0], axis=0)
        cholesky = np.linalg.cholesky(self.correlation)
        return volatility @ cholesky

    @classmethod
    def example(cls):
        return cls(
            mu=[0.05, 0.03], sigma=[0.20, 0.15], correlation=[[1.0, 0.5], [0.5, 1.0]]
        )
