"""Black Scholes Process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess
from pyesg.utils import to_array


class BlackScholesProcess(StochasticProcess):
    """
    Black Scholes process: dX = X*exp((μ - (1/2)*σ**2)dt + σdW)

    Examples
    --------
    >>> bsp = BlackScholesProcess(mu=0.05, sigma=0.2, dividend=0.01)
    >>> bsp.drift(10.0)
    array([0.02])
    >>> bsp.diffusion(10.0)
    array([0.2])
    >>> bsp.expectation(10.0, 0.5)
    array([10.10050167])
    >>> bsp.standard_deviation(10.0, 0.5)
    array([0.14142136])
    >>> bsp.step(10.0, dt=0.5, random_state=42)
    array([10.83553577])
    """

    def __init__(self, mu: float, sigma: float, dividend: float = 0.0) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.dividend = dividend

    def coefs(self) -> Dict[str, float]:
        return dict(mu=self.mu, sigma=self.sigma, dividend=self.dividend)

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return x0 * np.exp(dx)

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        return to_array(self.mu - self.dividend - 0.5 * self.sigma * self.sigma)

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        return to_array(self.sigma)


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
