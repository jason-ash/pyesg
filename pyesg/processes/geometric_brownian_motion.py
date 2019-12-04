"""Geometric Brownian Motion"""
from typing import Dict

from pyesg.processes import StochasticProcess, Vector


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion process: dX = μXdt + σXdW

    Examples
    --------
    >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    >>> gbm
    <pyesg.GeometricBrownianMotion{'mu': 0.05, 'sigma': 0.2}>
    >>> gbm.drift(x0=2.0)
    0.1
    >>> gbm.diffusion(x0=2.0)
    0.4
    >>> gbm.expectation(x0=1.0, dt=0.5)
    1.025
    >>> gbm.standard_deviation(x0=1.0, dt=0.5)
    0.14142135623730953
    >>> gbm.step(x0=1.0, dt=1.0, random_state=42)
    array([1.14934283])
    >>> gbm.step(x0=np.array([1.0, 15.0, 50.0]), dt=1.0, random_state=42)
    array([ 1.14934283, 15.3352071 , 58.97688538])
    >>> gbm.logpdf(x0=1.0, xt=1.1, dt=1.0)
    0.6592493792294276
    """

    def __init__(self, mu: Vector, sigma: Vector) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def coefs(self) -> Dict[str, Vector]:
        return dict(mu=self.mu, sigma=self.sigma)

    def drift(self, x0: Vector) -> Vector:
        return self.mu * x0

    def diffusion(self, x0: Vector) -> Vector:
        return self.sigma * x0


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
