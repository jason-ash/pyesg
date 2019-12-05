"""Heston stochastic volatility process"""
from typing import Dict
import numpy as np

from pyesg.processes import StochasticProcess, Vector


class HestonProcess(StochasticProcess):
    """
    Heston stochastic volatility process

    Parameters
    ----------
    mu : Union[float, np.ndarray], the expected rate of return of the underlying asset
    theta : Union[float, np.ndarray], the long-term expected volatility level
    kappa : Union[float, np.ndarray], the mean reversion speed of the variance
    sigma : Union[float, np.ndarray], the volatility-of-variance
    rho : np.ndarray, the correlation coefficient between the two stochastic processes
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, mu: Vector, theta: Vector, kappa: Vector, sigma: Vector, rho: Vector
    ) -> None:
        super().__init__()
        self.mu = mu
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho

    def coefs(self) -> Dict[str, Vector]:
        return dict(
            mu=self.mu,
            theta=self.theta,
            kappa=self.kappa,
            sigma=self.sigma,
            rho=self.rho,
        )

    def drift(self, x0: np.ndarray) -> np.ndarray:
        # floor volatility at zero
        x0[1] = max(0.0, x0[1])

        # drift of the underlying asset price and the variance, separately
        underlying = self.mu - x0[1] * x0[1] / 2
        variance = self.kappa * (self.theta - x0[1] * x0[1])
        return np.array([underlying, variance])

    def diffusion(self, x0: np.ndarray) -> np.ndarray:
        # floor volatility near zero, but positive, to keep correlation effect
        x0[1] = max(1e-6, x0[1])
        out = np.zeros(shape=(2, 2), dtype=np.float64)
        out[0, 0] = x0[1]
        out[0, 1] = 0.0
        out[1, 0] = self.rho * self.sigma * x0[1]
        out[1, 1] = self.sigma * x0[1] * (1 - self.rho * self.rho) ** 0.5
        return out


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
