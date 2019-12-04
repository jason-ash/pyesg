"""Classes for stochastic processes"""
from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen

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

    def __init__(self, dW: rv_continuous = stats.norm) -> None:
        self.dW = dW

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}{self.coefs()}>"

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self.coefs().values())

    @abstractmethod
    def coefs(self) -> Dict[str, Vector]:
        """Returns a dictionary of the process coefficients"""

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

    def transition_distribution(self, x0: Vector, dt: float) -> rv_frozen:
        """
        Returns a calibrated scipy.stats distribution object for the transition, given
        a starting value, x0
        """
        loc = self.expectation(x0=x0, dt=dt)
        scale = self.standard_deviation(x0=x0, dt=dt)
        return self.dW(loc=loc, scale=scale)

    def logpdf(self, x0: Vector, xt: Vector, dt: float) -> Vector:
        """
        Returns the log-probability of moving from x0 to x1 starting at time t and
        moving to time t + dt
        """
        return self.transition_distribution(x0=x0, dt=dt).logpdf(xt)

    def nnlf(self, x0: Vector, xt: Vector, dt: float) -> Vector:
        """
        Returns the negative log-likelihood function of moving from x0 to x1 starting at
        time t and moving to time t + dt
        """
        return -np.sum(self.logpdf(x0=x0, xt=xt, dt=dt))

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
    >>> ou.step(x0=np.array([0.03, 0.05, 0.09]), dt=1.0, random_state=42)
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
