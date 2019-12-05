"""Abstract base classes for pyesg stochastic processes"""
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

    def standard_deviation(self, x0: Vector, dt: float) -> np.ndarray:
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
        if isinstance(x0, list):
            x0 = np.array(x0, dtype=np.float64)
        dW = self.rvs(size=x0.shape, random_state=random_state)
        return (
            self.expectation(x0=x0, dt=dt) + self.standard_deviation(x0=x0, dt=dt) * dW
        )


class MultiStochasticProcess(StochasticProcess):  # pylint: disable=abstract-method
    """
    Abstract base class for a multi-stochastic diffusion process: a process that
    comprises at least two stochastic processes whose values depend on one another

    Parameters
    ----------
    correlation : np.ndarray, a square matrix of correlations among the stochastic
        portions of the processes. Its shape must match the number of processes
    dW : Scipy stats distribution object, default scipy.stats.norm. Specifies the
        distribution from which samples should be drawn.
    """

    def __init__(self, correlation: np.ndarray, dW: rv_continuous = stats.norm) -> None:
        super().__init__(dW=dW)
        self.correlation = correlation

    def step(self, x0: Vector, dt: float, random_state: RandomState = None) -> Vector:
        """
        Applies the stochastic process to an array of initial values using the Euler
        Discretization method
        """
        if isinstance(x0, (int, float)):
            x0 = np.array([x0], dtype=np.float64)
        if isinstance(x0, list):
            x0 = np.array(x0, dtype=np.float64)
        dW = self.rvs(size=x0[None, :].shape, random_state=random_state)
        return (
            self.expectation(x0=x0, dt=dt)
            + (dW @ self.standard_deviation(x0=x0, dt=dt).T).squeeze()
        )


class StochasticProcessArray(StochasticProcess):
    """
    Abstract base class for a stochastic diffusion process array: a process that
    comprises several correlated stochastic processes, given a correlation matrix, but
    whose values don't depend on one another

    Parameters
    ----------
    correlation : np.ndarray, a square matrix of correlations among the stochastic
        portions of the processes. Its shape must match the number of processes.
    """

    def __init__(self, correlation: np.ndarray, dW: rv_continuous = stats.norm) -> None:
        super().__init__(dW=dW)
        self.correlation = correlation

    @abstractproperty
    def processes(self) -> List[StochasticProcess]:
        """
        Returns a list of the underlying StochasticProcess objects that make up the
        joint stochastic process
        """

    def drift(self, x0: Vector) -> Vector:
        """Returns an array of the drifts of the underlying processes in order"""
        return np.array([p.drift(x0=x0) for p in self.processes])

    def diffusion(self, x0: Vector) -> Vector:
        """Returns an array of the diffusions of the underlying processes in order"""
        return np.array([p.diffusion(x0=x0) for p in self.processes])

    def rvs(
        self, size: Tuple[int, ...], random_state: RandomState = None
    ) -> np.ndarray:
        """
        Returns an array of correlated random numbers from the underlying distribution
        """
        cov = np.linalg.cholesky(self.correlation)
        rvs = self.dW.rvs(size=size, random_state=check_random_state(random_state))
        return rvs @ cov.T
