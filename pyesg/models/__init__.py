"""Base classes for models"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats

# from pyesg.models.samplers import recursive_sampler


class DiffusionProcess:
    """
    Base class for stochastic diffusion processes.

    Each diffusion process is callable, where the function represents the discrete
    implementation of the continuous diffusion process. That discrete implementation can
    be used to generate random samples from the distribution.

    Each subclass should implement an __init__ with the required parameters of that
    process; for example, mu and sigma for geometric brownian motion.
    """

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __call__(
        self, x: float, dt: float, dW: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Defines the discrete implementation of the DiffusionProcess. Takes a value of x,
        then simulates a single step in the process to generate a new value.

        Parameters
        ----------
        x : float, the current, or starting value of the process
        dt : float, the amount of time to simulate between x and the next value
        dW : Union[float, np.ndarray], a float or array of normal random variates

        Returns
        -------
        sample : np.ndarray, a randomly simulated value or array of values computed by
            the given diffusion process
        """
        raise NotImplementedError()

    @property
    def _fitted_params(self) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of fitted model parameters.
        Parameters should default to None if they haven't been fitted yet.
        """
        raise NotImplementedError()

    @property
    def _check_fitted(self) -> bool:
        """Returns a boolean indicating whether or not the model has been fitted"""
        return all(x is not None for x in self._fitted_params.values())

    def sample(
        self,
        n_scen: int,
        n_years: int,
        step_size: int,
        init: Union[float, np.ndarray],
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample from the diffusion process for a number of scenarios and time steps.

        Parameters
        ----------
        n_scen : int, the number of scenarios to generate
        n_years : int, the number of years per scenario
        step_size : int, the number of steps per year; e.g. 1 for annual time steps, 12
            for monthly, 24 for bi-weekly, 52 for weekly, 252 (or 365) for daily
        init : Union[float, np.ndarray], either a single start value that will be
            broadcast to all scenarios, or a start value array that should match the
            shape of "n_scen"
        random_state : Optional[int], to ensure repeated results if desired. If None,
            then results will be created with no random seed

        Returns
        -------
        scenarios : np.ndarray with shape (n_scen, 1 + n_years*step_size), with the scenario
            results from the process
        """
        if not self._check_fitted:
            raise RuntimeError("Must call 'fit' first!")

        # create an array of random numbers we'll need to generate the scenarios
        # then overwrite the first value of each scenario equal as "init"
        # this is currently assumed to be a normal distribution - come back
        # later to potentially add some flexibility to use other distributions
        scenarios = stats.norm.rvs(
            size=(n_scen, 1 + n_years * step_size), random_state=random_state
        )
        scenarios[:, 0] = init

        # recursive calls operate on rows, but we can parallelize over scenarios
        dt = 1 / step_size
        for i in range(n_years * step_size):
            scenarios[:, i + 1] = self(x=scenarios[:, i], dt=dt, dW=scenarios[:, i + 1])
        return scenarios
