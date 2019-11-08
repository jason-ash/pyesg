"""Base classes for models"""
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from pyesg.models.samplers import recursive_sampler


class DiffusionProcess:
    """Base class for stochastic diffusion processes"""

    def __repr__(self) -> str:
        return self.__class__.__name__

    def equation(self) -> Optional[float]:
        """
        Discretized formula to approximate the continuous solution to the
        stochastic differential equation for the given diffusion process.

        Should be defined for each model that subclasses DiffusionProcess
        """
        raise NotImplementedError

    def sample(self):
        """Sample from the diffusion process"""
        raise NotImplementedError
