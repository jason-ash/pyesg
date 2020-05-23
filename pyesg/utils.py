"""Useful functions"""
from typing import List, Union, Tuple
import numpy as np
import pandas as pd


# typing aliases
Array = Union[
    float,
    int,
    List[float],
    List[int],
    Tuple[float, ...],
    Tuple[int, ...],
    np.ndarray,
    pd.Series,
]
RandomState = Union[int, np.random.RandomState, None]


def _has_valid_cholesky(matrix: np.ndarray) -> bool:
    """Determine whether or not a matrix has a valid Cholesky decomposition"""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def check_random_state(seed: RandomState) -> np.random.RandomState:
    """
    Returns a numpy RandomState object from any of an integer, a RandomState object, or
    a None value (randomly instantiated RandomState object)
    """
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("Invalid argument type to convert to a RandomState object")


def to_array(value: Array) -> np.ndarray:
    """
    Converts an input value(s) to a numpy float64 array.

    Parameters
    ----------
    x0 : float, int, List[float], List[int], Tuple[float], Tuple[int], np.ndarray, an
        "array" of initial values to pass into one of the process methods

    Returns
    -------
    x0 : np.ndarray, a numpy array version of the original values passed

    Raises
    ------
    TypeError : if value is not one of the handled types

    Examples
    --------
    >>> to_array(10)
    array([10.])
    >>> to_array(15.0)
    array([15.])
    >>> to_array([10, 11])
    array([10., 11.])
    >>> to_array([15., 16.])
    array([15., 16.])
    >>> to_array([10, 11.5])
    array([10. , 11.5])
    >>> to_array((5))
    array([5.])
    >>> to_array((10, 11.5))
    array([10. , 11.5])
    >>> to_array(np.array([10, 12]))
    array([10., 12.])
    >>> to_array(np.array(5))
    array([5.])
    >>> to_array(np.array([0.3, 0.5, 0.7]))
    array([0.3, 0.5, 0.7])
    >>> to_array([[1.0, 0.5], [0.5, 1.0]])
    array([[1. , 0.5],
           [0.5, 1. ]])
    >>> to_array(pd.Series([0.3, 0.5, 0.7]))
    array([0.3, 0.5, 0.7])
    """
    if isinstance(value, (int, float)):
        return np.array([value], dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float64)
    if isinstance(value, pd.Series):
        return value.values
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.astype(np.float64)[None]
        return value.astype(np.float64)
    raise TypeError(f"{value} is not a valid type")
