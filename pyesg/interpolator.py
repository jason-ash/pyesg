"""Estimator classes for interest rate curve interpolation"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np


class Interpolator(ABC):
    """Abstract base class for Interpolators"""

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__qualname__}>"

    @abstractmethod
    def __call__(
        self, X: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        """Returns the Interpolator estimate of a rate at maturity X"""

    def is_fit(self) -> bool:
        """Returns a boolean indicating whether the model parameters have been fit"""
        return all(self.coefs().values())

    @abstractmethod
    def coefs(self) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of fitted model parameters.
        Parameters should default to None if they haven't been fitted yet.
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the interpolator using ordinary least squares

        Parameters
        ----------
        X : np.array of maturies, must be >0
        y : np.array of rates corresponding to each maturity

        Returns
        -------
        self : returns an instance of self
        """

    def predict(self, X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self.is_fit():
            return self(X, **self.coefs())
        raise RuntimeError("Must call 'fit' first!")
