"""Cox-Ingersoll-Ross Process"""
from typing import Dict

from pyesg.processes import StochasticProcess, Vector


class CoxIngersollRossProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross process: dX = θ(μ - X)dt + σX**0.5dW

    Examples
    --------
    >>> cir = CoxIngersollRossProcess(mu=0.05, sigma=0.02, theta=0.1)
    >>> cir
    <pyesg.CoxIngersollRossProcess{'mu': 0.05, 'sigma': 0.02, 'theta': 0.1}>
    >>> cir.drift(x0=0.045)
    0.0005000000000000004
    >>> cir.diffusion(x0=0.03)
    0.0034641016151377548
    >>> cir.expectation(x0=0.03, dt=0.5)
    0.031
    >>> cir.standard_deviation(x0=0.03, dt=0.5)
    0.0024494897427831783
    >>> cir.step(x0=0.03, dt=1.0, random_state=42)
    array([0.03372067])
    >>> cir.step(x0=[0.03, 0.05, 0.09], dt=1.0, random_state=42)
    array([0.03372067, 0.04938166, 0.08988613])
    >>> cir.logpdf(x0=0.05, xt=0.02, dt=1.0)
    -18.009049390999536
    """

    def __init__(self, mu: Vector, sigma: Vector, theta: Vector) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def coefs(self) -> Dict[str, Vector]:
        return dict(mu=self.mu, sigma=self.sigma, theta=self.theta)

    def drift(self, x0: Vector) -> Vector:
        return self.theta * (self.mu - x0)

    def diffusion(self, x0: Vector) -> Vector:
        return self.sigma * x0 ** 0.5


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
