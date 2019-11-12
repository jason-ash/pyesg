"""Classes for stochastic diffusion process"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats


class DiffusionProcess:
    """
    Base class for a stochastic diffusion process.

    Provides the framework for implementing specific stochastic models as subclasses,
    including a __call__ method that describes how to generate new samples from the
    process, given a start value and a delta-t.

    Parameters
    ----------
    n_indices : int, default 1, the number of indices to model. >1 is appropriate for
        some equity models, where we may wish to model correlated assets. Often would
        be set to 1 for short-rate interest models.
    """

    def __init__(self, n_indices: int = 1) -> None:
        self.n_indices = n_indices
        if n_indices > 1:
            self.correlation: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}{self._coefs}>"

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """
        Simulates the next value or array of values from the process,
        given a delta-t expressed in years, e.g. 1/12 for monthly. Can
        be deterministically drawn if a random_state is specified.

        Parameters
        ----------
        value : Union[float, np.ndarray], the starting value or array
        dt : float, the discrete time elapsed between value(t) and value(t+dt)
        random_state : Optional[int, np.random.RandomState], either an integer seed or a
            numpy RandomState object directly, if reproducibility is desired

        Returns
        -------
        samples : np.ndarray, the next values in the process
        """
        raise NotImplementedError()

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        """Returns a dictionary of parameters required for this process"""
        raise NotImplementedError()

    @property
    def _stochastic_dist(self):
        """
        Returns a scipy distribution rvs method that can be used to generate stochastic
        samples for new values in the diffusion process. Default is standard normal.
        """
        return stats.norm.rvs

    def _dW(
        self,
        size: Tuple[int, ...],
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """
        Returns a batch of random values for the process, given a size and optional
        random_state integer or object. If multiple indices are desired, then the
        correlation matrix is transformed using np.linalg.cholesky, then multiplied by
        the random variates to generate correlated draws.
        """
        rv = self._stochastic_dist(size=size, random_state=random_state)

        try:
            if hasattr(self, "correlation"):
                cholesky = np.linalg.cholesky(self.correlation)
                rv = rv @ cholesky.T
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError("Correlation matrix is not valid") from e
        return rv

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether or not the process has been fit"""
        return all([v is not None for v in self._coefs.values()])

    def _validate_coefs(self) -> None:
        """Validates shape, type, and ranges of coefficients for the process"""
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the parameters of the diffusion process based on historical data.

        The exact method of fitting should be defined at the subclass level, because the
        implementation can vary depending on the model.

        Parameters
        ----------
        X : np.ndarray, the indices of times/dates of the observed prices
        y : np.ndarray, the observed prices or values on the given dates. If multiple
            indices, then y will be a matrix, where the columns are the indices.

        Returns
        -------
        self
        """
        raise NotImplementedError()

    def sample(
        self,
        init: Union[float, np.ndarray],
        n_scen: int,
        n_year: int,
        n_step: int,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        """
        Sample from the diffusion process for a number of scenarios and time steps.

        Parameters
        ----------
        init : Union[float, np.ndarray], either a single start value that will be
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
        samples : np.ndarray, with the scenario results from the process
            if n_indices > 1, then shape = (n_scen, 1 + n_years * n_step, n_indices)
            otherwise, shape = (n_scen, 1 + n_years * n_step)
        """
        if not self._is_fit():
            raise RuntimeError("Model parameters haven't been fit yet!")

        # set a function-level pseudo random number generator, either by creating a new
        # RandomState object with the integer argument, or using the RandomState object
        # directly passed in the arguments.
        if isinstance(random_state, int):
            prng = np.random.RandomState(random_state)
        else:
            prng = random_state

        # create a shell array that we will populate with values once they are available
        samples = np.empty(shape=(n_scen, 1 + n_year * n_step, self.n_indices))

        # overwrite first value of each scenario (the first column) with the init value
        # confirm that if init is passed as an array that it matches the n_scen shape
        try:
            samples[:, 0, :] = init
        except ValueError as e:
            raise ValueError("'init' should have the same length as 'n_scen'") from e

        # generate the next value recursively, but vectorized across scenarios (columns)
        # also vectorized across indices, if applicable
        for i in range(n_year * n_step):
            samples[:, i + 1, :] = self(
                value=samples[:, i, :], dt=1 / n_step, random_state=prng
            )

        # squeeze the final dimension of the array to simplify if n_indices == 1
        return samples.squeeze()


class Vasicek(DiffusionProcess):
    """Vasicek short-rate model"""

    def __init__(self) -> None:
        super().__init__(n_indices=1)
        self.k: Optional[float] = None
        self.theta: Optional[float] = None
        self.sigma: Optional[float] = None

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        if isinstance(value, float):
            # convert to a one-element array
            value = np.array(value)
        dW = self._dW(size=value.shape, random_state=random_state)
        return value + self.k * (self.theta - value) * dt + self.sigma * dt ** 0.5 * dW

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        return dict(k=self.k, theta=self.theta, sigma=self.sigma)

    def _validate_coefs(self) -> None:
        raise NotImplementedError("TODO - not implemented yet.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("TODO - not implemented yet.")


class CoxIngersollRoss(DiffusionProcess):
    """Cox-Ingersoll-Ross short-rate model"""

    def __init__(self) -> None:
        super().__init__(n_indices=1)
        self.k: Optional[float] = None
        self.theta: Optional[float] = None
        self.sigma: Optional[float] = None

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        if isinstance(value, float):
            # convert to a one-element array
            value = np.array(value)
        dW = self._dW(size=value.shape, random_state=random_state)
        return (
            value
            + self.k * (self.theta - value) * dt
            + self.sigma * value ** 0.5 * dt ** 0.5 * dW
        )

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        return dict(k=self.k, theta=self.theta, sigma=self.sigma)

    def _validate_coefs(self) -> None:
        raise NotImplementedError("TODO - not implemented yet.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("TODO - not implemented yet.")


class GeometricBrownianMotion(DiffusionProcess):
    """Geometric Brownian Motion process"""

    def __init__(self, n_indices: int = 1) -> None:
        super().__init__(n_indices=n_indices)
        self.mu: Optional[float] = None
        self.sigma: Optional[float] = None

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> np.ndarray:
        if isinstance(value, float):
            # convert to a one-element array
            value = np.array(value)
        dW = self._dW(size=value.shape, random_state=random_state)
        return value * np.exp(
            (self.mu - self.sigma * self.sigma / 2) * dt + self.sigma * dt ** 0.5 * dW
        )

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        _coefs = dict(mu=self.mu, sigma=self.sigma)
        if self.n_indices > 1:
            _coefs["correlation"] = self.correlation
        return _coefs

    def _validate_coefs(self) -> None:
        raise NotImplementedError("TODO - not implemented yet.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("TODO - not implemented yet.")
