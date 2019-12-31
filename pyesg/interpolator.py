"""Estimator classes for interest rate curve interpolation"""
from typing import Dict, Optional, Union
import numpy as np


class Interpolator:
    """Base class for Interpolators"""

    def __repr__(self) -> str:
        return f"<pyesg.{self.__class__.__name__}>"

    def __call__(
        self, x: Union[float, np.ndarray], **params
    ) -> Union[float, np.ndarray]:
        """Returns the Interpolator estimate of a rate at maturity x"""
        raise NotImplementedError()

    @property
    def _fitted_params(self) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of fitted model parameters.
        Parameters should default to None if they haven't been fitted yet.
        """
        raise NotImplementedError()

    def _check_fitted(self) -> bool:
        """Returns a boolean indicating whether or not the model has been fitted"""
        return all(x is not None for x in self._fitted_params.values())

    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the predicted values from an array of independent values"""
        if self._check_fitted():
            return self(x, **self._fitted_params)
        raise RuntimeError("Must call 'fit' first!")
