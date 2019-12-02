"""pyesg models"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np
from scipy import stats

from pyesg.utils import check_random_state


class StochasticProcess(ABC):
    """
    Abstract base class for a stochastic diffusion process

    Parameters
    ----------
    dW : Scipy stats distribution object, default scipy.stats.norm. Specifies the
        distribution from which samples should be drawn.
    """

    def __init__(
        self,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        dim: int = 1,
        dW: stats.rv_continuous = stats.norm,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.dW = dW

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}{self._coefs()}>"

    @abstractmethod
    def __call__(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> np.ndarray:
        """
        Simulates the next value or array of values from the process, given a delta-t
        expressed in years, e.g. 1/12 for monthly. Can be deterministically drawn if a
        random_state is specified.

        Parameters
        ----------
        x0 : Union[float, np.ndarray], the starting value or array of values
        dt : float, the discrete time elapsed between value(t) and value(t+dt), in years
        random_state : Optional[int, np.random.RandomState], either an integer seed or a
            numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        samples : np.ndarray, the next values in the process
        """

    @property
    def mu(self) -> Optional[float]:
        """Returns the drift parameter of the process"""
        return self._mu

    @mu.setter
    def mu(self, value: Optional[float]) -> None:
        self._mu = value

    @property
    def sigma(self) -> Optional[float]:
        """Returns the volatility parameter of the process"""
        return self._sigma

    @sigma.setter
    def sigma(self, value: Optional[float]) -> None:
        if value:
            if value < 0.0:
                raise ValueError(f"{value} is not valid; sigma should be positive")
        self._sigma = value

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the stochastic process (dimension can be thought of as
        the number of processes being modeled.)
        """
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"{value} is not valid; dimension must be at least 1")
        self._dim = value

    @property
    def dW(self) -> stats.rv_continuous:
        """Returns the underlying random distribution for this process"""
        return self._dW

    @dW.setter
    def dW(self, value: stats.rv_continuous):
        if not isinstance(value, stats.rv_continuous):
            raise ValueError(f"{value} is not a valid scipy.stats distribution object")
        self._dW = value

    @abstractmethod
    def _coefs(self) -> Dict[str, Union[float, np.ndarray, None]]:
        """Returns a dictionary of parameters required for this process"""

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self._coefs().values())

    def sample(
        self,
        x0: Union[float, np.ndarray],
        n_scen: int,
        n_years: int,
        n_step: int,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> np.ndarray:
        """
        Sample from the diffusion process for a number of scenarios and time steps.

        Parameters
        ----------
        x0 : Union[float, np.ndarray], either a single start value that will be
            broadcast to all scenarios, or a start value array that should match the
            shape of "n_scen"
        n_scen : int, the number of scenarios to generate
        n_year : int, the number of years per scenario
        n_step : int, the number of steps per year; e.g. 1 for annual time steps, 12
            for monthly, 24 for bi-weekly, 52 for weekly, 252 (or 365) for daily
        random_state : Optional[int, np.random.RandomState], either an integer seed or a
            numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        samples : np.ndarray, with the scenario results from the process, with shape
            (n_scen, 1 + n_years * n_step)
        """
        if not self._is_fit():
            raise RuntimeError("Model parameters haven't been fit yet!")

        # create a shell array that we will populate with values once they are available
        size = (n_scen, 1 + n_years * n_step)
        samples = np.empty(shape=size)

        # overwrite first value of each scenario (the first column) with the init value
        # confirm that if init is passed as an array that it matches the n_scen shape
        try:
            samples[:, 0] = x0
        except ValueError as e:
            raise ValueError("'x0' should have the same length as 'n_scen'") from e

        # set a function-level pseudo random number generator, either by creating a new
        # RandomState object with the integer argument, or using the RandomState object
        # directly passed in the arguments.
        prng = check_random_state(random_state)

        # generate the next value recursively, but vectorized across scenarios (columns)
        for i in range(n_years * n_step):
            samples[:, i + 1] = self(x0=samples[:, i], dt=1 / n_step, random_state=prng)
        return samples


class WienerProcess(StochasticProcess):
    """
    Generalized Wiener process: dX = μdt + σdW

    Examples
    --------
    >>> wp = WienerProcess()
    >>> wp
    <pyesg.WienerProcess{'mu': None, 'sigma': None}>
    >>> wp._coefs()
    {'mu': None, 'sigma': None}
    >>> wp.sigma = -0.20
    Traceback (most recent call last):
     ...
    ValueError: -0.2 is not valid; sigma should be positive
    >>> wp.mu, wp.sigma = 0.05, 0.20
    >>> wp(np.full(5, 0.0), 1.0, random_state=42)
    array([0.14934283, 0.02234714, 0.17953771, 0.35460597, 0.00316933])
    >>> wp.sample(0.0, 5, 4, 1, random_state=42)
    array([[ 0.        ,  0.14934283,  0.15251544,  0.1098319 ,  0.04737439],
           [ 0.        ,  0.02234714,  0.3881897 ,  0.34504375,  0.19247753],
           [ 0.        ,  0.17953771,  0.38302465,  0.48141711,  0.59426657],
           [ 0.        ,  0.35460597,  0.31071109, -0.02194495, -0.15354977],
           [ 0.        ,  0.00316933,  0.16168133, -0.13330223, -0.36576297]])
    >>> wp.sample(np.arange(5), 5, 4, 1, random_state=42)
    array([[0.        , 0.14934283, 0.15251544, 0.1098319 , 0.04737439],
           [1.        , 1.02234714, 1.3881897 , 1.34504375, 1.19247753],
           [2.        , 2.17953771, 2.38302465, 2.48141711, 2.59426657],
           [3.        , 3.35460597, 3.31071109, 2.97805505, 2.84645023],
           [4.        , 4.00316933, 4.16168133, 3.86669777, 3.63423703]])
    """

    def __init__(
        self, mu: Optional[float] = None, sigma: Optional[float] = None, dim: int = 1
    ) -> None:
        super().__init__(mu=mu, sigma=sigma, dim=dim)

    def __call__(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> np.ndarray:
        # if necessary, convert x0 to an array before starting
        if isinstance(x0, float):
            x0 = np.array([x0])
        rvs = self.dW.rvs(size=x0.shape, random_state=random_state)
        return x0 + self.mu * dt + self.sigma * dt ** 0.5 * rvs

    def _coefs(self) -> Dict[str, Union[float, np.ndarray, None]]:
        return dict(mu=self.mu, sigma=self.sigma)


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Ornstein Uhlenbeck process: dX = θ * (μ - X)dt + σdW

    Examples
    --------
    >>> ou = OrnsteinUhlenbeckProcess()
    >>> ou
    <pyesg.OrnsteinUhlenbeckProcess{'mu': None, 'sigma': None, 'theta': None}>
    >>> ou._coefs()
    {'mu': None, 'sigma': None, 'theta': None}
    >>> ou.theta = -0.20
    Traceback (most recent call last):
     ...
    ValueError: -0.2 is not valid; theta should be positive
    >>> ou.mu, ou.sigma, ou.theta = 0.045, 0.015, 0.15
    >>> ou.sample(0.03, 5, 4, 1, random_state=42)
    array([[ 0.03      ,  0.03970071,  0.03698355,  0.03123475,  0.02486523],
           [ 0.03      ,  0.03017604,  0.05608782,  0.0474387 ,  0.03188043],
           [ 0.03      ,  0.04196533,  0.05393205,  0.05622168,  0.05925213],
           [ 0.03      ,  0.05509545,  0.04653901,  0.01760896,  0.00809725],
           [ 0.03      ,  0.0287377 ,  0.03931545,  0.01429436, -0.00228435]])
    """

    def __init__(
        self,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        theta: Optional[float] = None,
        dim: int = 1,
    ) -> None:
        super().__init__(mu=mu, sigma=sigma, dim=dim)
        self.theta = theta

    def __call__(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> np.ndarray:
        # if necessary, convert x0 to an array before starting
        if isinstance(x0, float):
            x0 = np.array([x0])
        rvs = self.dW.rvs(size=x0.shape, random_state=random_state)
        return x0 + self.theta * (self.mu - x0) * dt + self.sigma * dt ** 0.5 * rvs

    @property
    def theta(self) -> Optional[float]:
        """Returns the mean-reversion parameter of the process"""
        return self._theta

    @theta.setter
    def theta(self, value: Optional[float]):
        if value:
            if value < 0.0:
                raise ValueError(f"{value} is not valid; theta should be positive")
        self._theta = value

    def _coefs(self) -> Dict[str, Union[float, np.ndarray, None]]:
        return dict(mu=self.mu, sigma=self.sigma, theta=self.theta)


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion process: dX = μXdt + σXdW

    Examples
    --------
    >>> gbm = GeometricBrownianMotion()
    >>> gbm
    <pyesg.GeometricBrownianMotion{'mu': None, 'sigma': None}>
    >>> gbm._coefs()
    {'mu': None, 'sigma': None}
    >>> gbm.sigma = -0.20
    Traceback (most recent call last):
     ...
    ValueError: -0.2 is not valid; sigma should be positive
    >>> gbm.mu, gbm.sigma = 0.045, 0.15
    >>> gbm.sample(10.0, 5, 4, 1, random_state=42)
    array([[10.        , 11.14334229, 11.12808032, 10.73714278, 10.20743003],
           [10.        , 10.13095358, 13.27957805, 12.80861392, 11.38097656],
           [10.        , 11.3985753 , 13.2282353 , 14.18801968, 15.38333874],
           [10.        , 12.99792295, 12.52990784,  9.72671649,  8.77952903],
           [10.        ,  9.98627936, 11.20484298,  8.94735912,  7.4877089 ]])
    """

    def __init__(
        self, mu: Optional[float] = None, sigma: Optional[float] = None, dim: int = 1
    ) -> None:
        super().__init__(mu=mu, sigma=sigma, dim=dim)

    def __call__(
        self,
        x0: Union[float, np.ndarray],
        dt: float,
        random_state: Union[int, np.random.RandomState, None] = None,
    ) -> np.ndarray:
        # if necessary, convert x0 to an array before starting
        if isinstance(x0, float):
            x0 = np.array([x0])
        rvs = self.dW.rvs(size=x0.shape, random_state=random_state)
        return x0 * np.exp(
            (self.mu - self.sigma * self.sigma / 2) * dt + self.sigma * dt ** 0.5 * rvs
        )

    def _coefs(self) -> Dict[str, Union[float, np.ndarray, None]]:
        return dict(mu=self.mu, sigma=self.sigma)


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
