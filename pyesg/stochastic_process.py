"""Abstract base classes for pyesg stochastic processes"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous

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

    In addition, each subclass should define a classmethod called `example`, which will
    instantiate a model with reasonable parameters for users to be able to get started.

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
        self.dW = dW
        self.dim = dim

    def __repr__(self) -> str:
        """Returns a string representation of this model"""
        params = (f"{k}={repr(v)}" for k, v in self.coefs().items())
        return f"<pyesg.{self.__class__.__qualname__}({', '.join(params)})>"

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

    @classmethod
    @abstractmethod
    def example(cls):  # creates an instance of the class with default parameters
        """
        Creates an instance of this model with sensible default parameters, primarily to
        be able to visualize or understand the dynamics of the model quickly.
        """

    def apply(self, x0: Array, dx: Array) -> np.ndarray:
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

    def rvs(
        self, n_scenarios: int, n_steps: int, random_state: RandomState = None
    ) -> np.ndarray:
        """
        Returns the array of random variates used to generate a batch of scenarios with
        shape (n_scenarios, n_steps, self.dim). If dim == 1, then the third dimension
        will be squeezed, so the returned array will have shape (n_scenarios, n_steps).

        Parameters
        ----------
        n_scenarios : int, the number of scenarios to generate, e.g. 1000
        n_steps : int, the number of steps in the scenario, e.g. 52
        random_state : Union[int, np.random.RandomState, None], either an integer seed
            or a numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        rvs : np.ndarray, an array of the random variates used to generate scenarios
        """
        rvs = np.zeros(shape=(n_scenarios, n_steps, self.dim))
        for i in range(n_steps):
            random_state = check_random_state(random_state)
            rvs[:, i, :] = self.dW.rvs(
                size=(n_scenarios, self.dim), random_state=random_state
            )
        return rvs.squeeze()

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
        if self.dim == 1:
            dx = rvs * self.standard_deviation(x0=x0, dt=dt)
        else:
            if x0.ndim == 1:
                # single sample from a joint process
                dx = rvs @ self.standard_deviation(x0=x0, dt=dt).transpose(1, 0)
            else:
                # multiple samples from a joint process
                # we have rvs as a (samples, dimension) array and standard deviation as
                # a (samples, dimension, dimension) array. We want to matrix multiply
                # the rvs (dimension) index with the transposed (dimension, dimension)
                # standard deviation for each sample to get a (samples, dimension) array
                dx = np.einsum("ab,acb->ac", rvs, self.standard_deviation(x0=x0, dt=dt))
        return self.apply(self.expectation(x0=x0, dt=dt), dx)

    def scenarios(  # pylint: disable=too-many-arguments
        self,
        x0: Array,
        dt: float,
        n_scenarios: int,
        n_steps: int,
        random_state: RandomState = None,
    ) -> np.ndarray:
        """
        Returns a recursively-generated scenario, starting with initial values/array, x0
        and continuing by steps with length dt for a given number of steps

        Parameters
        ----------
        x0 : Array, either a single start value or array of start values if applicable
        dt : float, the length between steps, in years, e.g. 1/12 for monthly steps
        n_scenarios : int, the number of scenarios to generate, e.g. 1000
        n_steps : int, the number of steps in the scenario, e.g. 52. In combination with
            dt, this determines the scope of the scenario, e.g. dt=1/12 and n_step=360
            will produce 360 monthly time steps, i.e. a 30-year monthly projection
        random_state : Union[int, np.random.RandomState, None], either an integer seed
            or a numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        samples : np.ndarray with shape (n_scenarios, n_steps + 1) for a one-dimensional
            stochastic process, or (n_scenarios, n_steps + 1, dim) for a two-dimensional
            stochastic process, where the first timestep of each scenario is x0
        """
        # set a function-level pseudo random number generator, either by creating a new
        # RandomState object with the integer argument, or using the RandomState object
        # directly passed in the arguments.
        prng = check_random_state(random_state)

        x0 = to_array(x0)  # ensure we're working with a numpy array before proceeding

        # create a shell array that we will populate with values once they are available
        # this is generally faster than appending subsequent steps to an array each time
        # we'll generate a 2d array if this process has dim == 1; otherwise it will be 3
        shape: Tuple[int, ...] = (n_scenarios, n_steps + 1)
        if self.dim > 1:
            shape = (shape[0], shape[1], self.dim)
        samples = np.empty(shape=shape, dtype=np.float64)

        try:
            # can we broadcast the x0 array into the number of scenarios we want?
            samples[:, 0] = x0
        except ValueError:
            raise ValueError(
                f"Could not broadcast the input array, with shape {x0.shape}, into "
                f"the scenario output array, with shape {samples.shape}"
            )

        # then we iterate through scenarios along the timesteps dimension
        for i in range(n_steps):
            samples[:, i + 1] = self.step(x0=samples[:, i], dt=dt, random_state=prng)

        return samples
