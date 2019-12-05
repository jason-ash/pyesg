"""Ornstein-Uhlenbeck Process"""
from typing import Dict

from pyesg.processes import StochasticProcess, Vector


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Generalized Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW

    Examples
    --------
    >>> ou = OrnsteinUhlenbeckProcess(mu=0.05, sigma=0.015, theta=0.15)
    >>> ou
    <pyesg.OrnsteinUhlenbeckProcess{'mu': 0.05, 'sigma': 0.015, 'theta': 0.15}>
    >>> ou.drift(x0=0.05)
    0.0
    >>> ou.diffusion(x0=0.03)
    0.015
    >>> ou.expectation(x0=0.03, dt=0.5)
    0.0315
    >>> ou.standard_deviation(x0=0.03, dt=0.5)
    0.010606601717798213
    >>> ou.step(x0=0.03, dt=1.0, random_state=42)
    array([0.04045071])
    >>> ou.step(x0=[0.03, 0.05, 0.09], dt=1.0, random_state=42)
    array([0.04045071, 0.04792604, 0.09371533])
    >>> ou.logpdf(x0=0.05, xt=0.09, dt=1.0)
    -0.2747890108803004
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
        # diffusion of an Ornstein-Uhlenbeck process does not depend on x0
        return self.sigma


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
