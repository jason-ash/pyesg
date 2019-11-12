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


def check_random_state(
    seed: Union[int, np.random.RandomState, None]
) -> np.random.RandomState:
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
