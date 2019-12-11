"""Abstract base classes for pyesg stochastic processes"""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen

from pyesg.utils import check_random_state, to_array, Array, RandomState


class StochasticProcess(ABC):
    """
    Abstract base class for a stochastic diffusion process. A stochastic processes can
    model any number of underlying variables, where the number of variables is defined
    by the "dim" attribute. Subclasses of StochasticProcess should define four methods:
        1. _drift : the drift component of the diffusion process; determines how much
            the process will move in the absence of any stochastic component
        2. _diffusion : the stochastic component of the diffusion process; determines
            how large the perturbations of the process will be
        3. _apply : instructions for how to update an initial value(s), given a vector
            of changes; e.g. addition or exponentiation
        4. coefs : a convenience method that stores all model coefficients in a dict so
            they can be referenced easily as a group

    Given these methods above, the base class provides methods for expected value of the
    process, standard deviation, and a transition density (if applicable). Also provides
    a method, "step", that iterates an initial vector of parameters one step forward.

    Parameters
    ----------
    dim : int, the dimension of the process; single-variable processes will have dim=1,
        while joint processes can have dim>1
    dW : Scipy stats distribution object, default scipy.stats.norm. Specifies the
        distribution from which samples should be drawn.
    """

    def __init__(self, dim: int = 1, dW: rv_continuous = stats.norm) -> None:
        self.dim = dim
        self.dW = dW

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}{self.coefs()}>"

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self.coefs().values())

    @abstractmethod
    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """Returns a new array of x-values, given a starting array and change vector"""

    @abstractmethod
    def _drift(self, x0: np.ndarray) -> np.ndarray:
        """Returns the drift component of the stochastic process"""

    @abstractmethod
    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        """Returns the diffusion component of the stochastic process"""

    @abstractmethod
    def coefs(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary of the process coefficients"""

    def apply(self, x0: Array, dx: np.ndarray) -> np.ndarray:
        """Returns a new array of x-values, given a starting array and change vector"""
        return self._apply(x0=to_array(x0), dx=to_array(dx))

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
        return self.apply(to_array(x0), self.drift(x0=x0) * dt)

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
        # generate an array of independent draws from the dW distribution (defaults to a
        # normal distribution.) In the general case, we can use matrix multiplication to
        # combine the random draws with the StochasticProcess's standard deviation. This
        # means that we can handle both single-dimensional and multi-dimensional
        # stochastic processes with a single abstract base class. For joint stochastic
        # processes, the standard deviation is a n x n matrix, where n is the dimension
        # of the process, so we effectively convert the independent random draws into
        # correlated random draws.
        x0 = to_array(x0)
        rvs = self.dW.rvs(size=x0.shape, random_state=check_random_state(random_state))
        dx = rvs @ self.standard_deviation(x0=x0, dt=dt).T
        return self.apply(self.expectation(x0=x0, dt=dt), dx)

    def scenario(
        self, x0: Array, dt: float, n_step: int, random_state: RandomState = None
    ) -> np.ndarray:
        """
        Returns a recursively-generated scenario, starting with initial values/array, x0
        and continuing by steps with length dt for a given number of steps

        Parameters
        ----------
        x0 : Array, either a single start value or array of start values if applicable
        dt : float, the length between steps
        n_step : int, the number of steps in the scenario, e.g. 360. In combination with
            dt, this determines the scope of the scenario, e.g. dt=1/12 and n_step=360
            will produce 360 monthly time steps, i.e. a 30-year monthly projection.
        random_state : Union[int, np.random.RandomState, None], either an integer seed
            or a numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        samples : np.ndarray with shape (n_step + 1, dim), where samples[0] is the input
            array, x0, and the subsequent indices are the steps of the scenario
        """
        # set a function-level pseudo random number generator, either by creating a new
        # RandomState object with the integer argument, or using the RandomState object
        # directly passed in the arguments.
        prng = check_random_state(random_state)

        # create a shell array that we will populate with values once they are available
        # this is generally faster than appending subsequent steps to an array each time
        samples = np.empty(shape=(n_step + 1, self.dim), dtype=np.float64)
        samples[0] = to_array(x0)
        for i in range(n_step):
            samples[i + 1] = self.step(x0=samples[i], dt=dt, random_state=prng)

        # squeeze the final dimension of the array if dim == 1
        return samples.squeeze()
