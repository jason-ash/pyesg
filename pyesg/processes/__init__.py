"""Abstract base classes for pyesg stochastic processes"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen

from pyesg.utils import check_random_state, to_array, Array, RandomState


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
    def _drift(self, x0: np.ndarray) -> np.ndarray:
        """Returns the drift component of the stochastic process"""

    @abstractmethod
    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        """Returns the diffusion component of the stochastic process"""

    @abstractmethod
    def coefs(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of the process coefficients"""

    def drift(self, x0: Array) -> np.ndarray:
        """Returns the drift component of the stochastic process"""
        return self._drift(x0=to_array(x0))

    def diffusion(self, x0: Array) -> np.ndarray:
        """Returns the diffusion component of the stochastic process"""
        return self._diffusion(x0=to_array(x0))

    def expectation(self, x0: Array, dt: float) -> np.ndarray:
        """
        Returns the expected value of the stochastic process using the Euler
        Discretization method
        """
        return to_array(x0) + self.drift(x0=x0) * dt

    def standard_deviation(self, x0: Array, dt: float) -> np.ndarray:
        """
        Returns the standard deviation of the stochastic process using the Euler
        Discretization method
        """
        return self.diffusion(x0=x0) * dt ** 0.5

    def transition_distribution(self, x0: Array, dt: float) -> rv_frozen:
        """
        Returns a calibrated scipy.stats distribution object for the transition, given
        a starting value, x0
        """
        loc = self.expectation(x0=x0, dt=dt)
        scale = self.standard_deviation(x0=x0, dt=dt)
        return self.dW(loc=loc, scale=scale)

    def logpdf(self, x0: Array, xt: Array, dt: float) -> np.ndarray:
        """
        Returns the log-probability of moving from x0 to x1 starting at time t and
        moving to time t + dt
        """
        return self.transition_distribution(x0=to_array(x0), dt=dt).logpdf(xt)

    def nnlf(self, x0: Array, xt: Array, dt: float) -> np.ndarray:
        """
        Returns the negative log-likelihood function of moving from x0 to x1 starting at
        time t and moving to time t + dt
        """
        return -np.sum(self.logpdf(x0=to_array(x0), xt=to_array(xt), dt=dt))

    def step(
        self, x0: Array, dt: float, random_state: RandomState = None
    ) -> np.ndarray:
        """
        Applies the stochastic process to an array of initial values using the Euler
        Discretization method
        """
        x0 = to_array(x0)
        rvs = self.dW.rvs(size=x0.shape, random_state=check_random_state(random_state))
        out = self.expectation(x0=x0, dt=dt) + rvs * self.standard_deviation(
            x0=x0, dt=dt
        )
        return to_array(out)


class JointStochasticProcess(StochasticProcess):  # pylint: disable=abstract-method
    """
    Abstract base class for a joint stochastic diffusion process: a process that
    comprises at least two correlated stochastic processes whose values may or may not
    depend on one another

    Parameters
    ----------
    correlation : np.ndarray, a square matrix of correlations among the stochastic
        portions of the processes. Its shape must match the number of processes
    dW : Scipy stats distribution object, default scipy.stats.norm. Specifies the
        distribution from which samples should be drawn.
    """

    def __init__(self, dW: rv_continuous = stats.norm) -> None:
        super().__init__(dW=dW)

    def step(
        self, x0: Array, dt: float, random_state: RandomState = None
    ) -> np.ndarray:
        """
        Applies the stochastic process to an array of initial values using the Euler
        Discretization method
        """
        x0 = to_array(x0)
        rvs = self.dW.rvs(
            size=x0[None, :].shape, random_state=check_random_state(random_state)
        )
        return (
            self.expectation(x0=x0, dt=dt)
            + (rvs @ self.standard_deviation(x0=x0, dt=dt).T).squeeze()
        )
