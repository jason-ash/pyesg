"""Classes for stochastic diffusion process"""
from typing import Dict, Optional, Union
import numpy as np
from scipy import stats


# pylint: disable=too-few-public-methods
class DiffusionProcess:
    """
    Base class for a stochastic diffusion process.

    Provides the framework for implementing specific stochastic models as subclasses,
    including a __call__ method that describes how to generate new samples from the
    process, given a start value and a delta-t.
    """

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulates the next value or array of values from the process,
        given a delta-t expressed in years, e.g. 1/12 for monthly. Can
        be deterministically drawn if a random_state is specified.

        Parameters
        ----------
        value : Union[float, np.ndarray], the starting value or array
        dt : float, the discrete time elapsed between value(t) and value(t+dt)
        random_seed : Optional[int], if reproducibility is desired

        Returns
        -------
        samples : np.ndarray, the next values in the process
        """
        raise NotImplementedError()

    @property
    def _stochastic_dist(self):
        """
        Returns a scipy distribution rvs method that can be used to generate stochastic
        samples for new values in the diffusion process. Default is standard normal.
        """
        return stats.norm.rvs

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        """Returns a dictionary of parameters required for this process"""
        raise NotImplementedError()

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether or not the process has been fit"""
        return all([v is not None for v in self._coefs.values()])


class Vasicek(DiffusionProcess):
    """Implements the Vasicek short-rate model"""

    def __init__(self) -> None:
        self.k: Optional[float] = None
        self.theta: Optional[float] = None
        self.sigma: Optional[float] = None

    def __call__(
        self,
        value: Union[float, np.ndarray],
        dt: float,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        pass

    @property
    def _coefs(self) -> Dict[str, Optional[float]]:
        return dict(k=self.k, theta=self.theta, sigma=self.sigma)


if __name__ == "__main__":
    V = Vasicek()
    V.k, V.theta, V.sigma = 0.15, 0.045, 0.015
    print(V._is_fit())
