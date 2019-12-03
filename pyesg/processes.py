"""Classes for stochastic processes"""
from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np
from scipy import stats

from pyesg.utils import check_random_state


# typing aliases
Vector = Union[float, np.ndarray]
RandomState = Union[int, np.random.RandomState, None]


class StochasticProcess(ABC):
    """
    Abstract base class for a stochastic diffusion process

    Parameters
    ----------
    dW : Scipy stats distribution object, default scipy.stats.norm. Specifies the
        distribution from which samples should be drawn.
    """

    def __init__(self, dW: stats.rv_continuous = stats.norm(0, 1)) -> None:
        self.dW = dW

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}{self.coefs()}>"

    @abstractmethod
    def coefs(self) -> Dict[str, Vector]:
        """Returns a dictionary of the process coefficients"""

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self.coefs().values())

    @abstractmethod
    def drift(self, x0: Vector) -> Vector:
        """Returns the drift component of the stochastic process"""

    @abstractmethod
    def diffusion(self, x0: Vector) -> Vector:
        """Returns the diffusion component of the stochastic process"""

    def expectation(self, x0: Vector, dt: float) -> Vector:
        """
        Returns the expected value of the stochastic process using the Euler
        Discretization method
        """
        return x0 + self.drift(x0=x0) * dt

    def standard_deviation(self, x0: Vector, dt: float) -> Vector:
        """
        Returns the standard deviation of the stochastic process using the Euler
        Discretization method
        """
        return self.diffusion(x0=x0) * dt ** 0.5

    def step(self, x0: Vector, dt: float, random_state: RandomState = None) -> Vector:
        """
        Applies the stochastic process to an array of initial values using the Euler
        Discretization method
        """
        if isinstance(x0, (int, float)):
            x0 = np.array([x0], dtype=np.float64)
        rvs = self.dW.rvs(size=x0.shape, random_state=check_random_state(random_state))
        return (
            self.expectation(x0=x0, dt=dt) + self.standard_deviation(x0=x0, dt=dt) * rvs
        )


class WienerProcess(StochasticProcess):
    """
    Generalized Wiener process: dX = μdt + σdW

    Examples
    --------
    >>> wp = WienerProcess(mu=0.05, sigma=0.2)
    >>> wp
    <pyesg.WienerProcess{'mu': 0.05, 'sigma': 0.2}>
    >>> wp.drift(x0=0.0)
    0.05
    >>> wp.diffusion(x0=0.0)
    0.2
    >>> wp.expectation(x0=0.0, dt=0.5)
    0.025
    >>> wp.standard_deviation(x0=0.0, dt=0.5)
    0.14142135623730953
    >>> wp.step(x0=0.0, dt=1.0, random_state=42)
    array([0.14934283])
    >>> wp.step(x0=np.array([0.0, 1.0, 2.0]), dt=1.0, random_state=42)
    array([0.14934283, 1.02234714, 2.17953771])
    """

    def __init__(self, mu: Vector, sigma: Vector) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def coefs(self) -> Dict[str, Vector]:
        return dict(mu=self.mu, sigma=self.sigma)

    def drift(self, x0: Vector) -> Vector:
        # drift of a Wiener process does not depend on x0
        return self.mu

    def diffusion(self, x0: Vector) -> Vector:
        # diffusion of a Wiener process does not depend on x0
        return self.sigma


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
