"""Estimator classes for interest rate curve interpolation"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np


class Interpolator(ABC):
    """Base class for Interpolators"""

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}>"

    @abstractmethod
    def __call__(
        self, x: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        """Returns the Interpolator estimate of a rate at maturity x"""

    def _is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self.coefs().values())

    @abstractmethod
    def coefs(self) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of fitted model parameters.
        Parameters should default to None if they haven't been fitted yet.
        """

    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self._is_fit():
            return self(x, **self.coefs())
        raise RuntimeError("Must call 'fit' first!")
