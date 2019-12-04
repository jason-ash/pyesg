"""Classes for stochastic processes"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Union
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

    def rvs(
        self, size: Tuple[int, ...], random_state: RandomState = None
    ) -> np.ndarray:
        """Returns an array of random numbers from the underlying distribution"""
        return self.dW.rvs(size=size, random_state=check_random_state(random_state))

    def step(self, x0: Vector, dt: float, random_state: RandomState = None) -> Vector:
        """
        Applies the stochastic process to an array of initial values using the Euler
        Discretization method
        """
        if isinstance(x0, (int, float)):
            x0 = np.array([x0], dtype=np.float64)
        dW = self.rvs(size=x0.shape, random_state=random_state)
        return (
            self.expectation(x0=x0, dt=dt) + self.standard_deviation(x0=x0, dt=dt) * dW
        )


class JointStochasticProcess(StochasticProcess, ABC):
    """
    Abstract base class for a joint stochastic diffusion process: a process that
    comprises several correlated stochastic processes, given a correlation matrix

    Parameters
    ----------
    correlation : np.ndarray, a square matrix of correlations among the stochastic
        portions of the processes. Its shape must match the number of processes.
    """

    def __init__(self, correlation: np.ndarray, dW: rv_continuous = stats.norm) -> None:
        super().__init__(dW=dW)
        self.correlation = correlation

    @abstractproperty
    def _processes(self) -> List[StochasticProcess]:
        """
        Returns a list of the underlying StochasticProcess objects that make up the
        joint stochastic process
        """

    def rvs(
        self, size: Tuple[int, ...], random_state: RandomState = None
    ) -> np.ndarray:
        """
        Returns an array of correlated random numbers from the underlying distribution
        """
        cov = np.linalg.cholesky(self.correlation)
        rvs = self.dW.rvs(size=size, random_state=check_random_state(random_state))
        return rvs @ cov.T


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
    def _processes(self):
        return [WienerProcess(mu=m, sigma=s) for m, s in zip(self.mu, self.sigma)]

    def coefs(self) -> Dict[str, Vector]:
        return dict(mu=self.mu, sigma=self.sigma)

    def drift(self, x0: Vector) -> Vector:
        return np.array([p.drift(x0=x0) for p in self._processes])

    def diffusion(self, x0: Vector) -> Vector:
        return np.array([p.diffusion(x0=x0) for p in self._processes])


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
