"""Useful functions"""
from typing import Union
import numpy as np


def _has_valid_cholesky(matrix: np.ndarray) -> bool:
    """Determine whether or not a matrix has a valid Cholesky decomposition"""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def _random_state_object(
    value: Union[int, np.random.RandomState, None]
) -> np.random.RandomState:
    """
    Returns a numpy RandomState object from any of an integer, a RandomState object, or
    a None value (randomly instantiated RandomState object)
    """
    if value is None:
        return np.random.RandomState()
    if isinstance(value, int):
        return np.random.RandomState(value)
    if isinstance(value, np.random.RandomState):
        return value
    raise ValueError("Invalid argument type to convert to a RandomState object")
