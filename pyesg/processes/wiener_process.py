"""Wiener Process"""
from typing import Dict
import numpy as np

from pyesg.processes import JointStochasticProcess, StochasticProcess, Vector


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


class JointWienerProcess(JointStochasticProcess):
    """
    Joint Wiener processes: dX = μdt + σdW

    Examples
    --------
    >>> jwp = JointWienerProcess(
    ...     mu=[0.05, 0.03], sigma=[0.20, 0.15], correlation=[[1.0, 0.5], [0.5, 1.0]]
    ... )
    >>> jwp
    <pyesg.JointWienerProcess{'mu': [0.05, 0.03], 'sigma': [0.2, 0.15]}>
    >>> jwp.drift(x0=[1.0, 1.0])
    array([0.05, 0.03])
    >>> jwp.diffusion(x0=[1.0, 1.0])
    array([0.2 , 0.15])
    >>> jwp.expectation(x0=[1.0, 1.0], dt=0.5)
    array([1.025, 1.015])
    >>> jwp.standard_deviation(x0=[1.0, 2.0], dt=2.0)
    array([0.28284271, 0.21213203])
    >>> jwp.step(x0=np.array([1.0, 1.0]), dt=1.0, random_state=42)
    array([1.14934283, 1.0492925 ])
    >>> jwp.correlation = [[1.0, 0.99], [0.99, 1.0]]
    >>> jwp.step(x0=np.array([1.0, 1.0]), dt=1.0, random_state=42)
    array([1.14934283, 1.10083636])
    """

    def __init__(self, mu: Vector, sigma: Vector, correlation: np.ndarray) -> None:
        super().__init__(correlation=correlation)
        self.mu = mu
        self.sigma = sigma

    @property
    def processes(self):
        return [WienerProcess(mu=m, sigma=s) for m, s in zip(self.mu, self.sigma)]

    def coefs(self) -> Dict[str, Vector]:
        return dict(mu=self.mu, sigma=self.sigma)


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
