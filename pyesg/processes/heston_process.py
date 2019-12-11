"""Heston stochastic volatility process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import JointStochasticProcess


class HestonProcess(JointStochasticProcess):
    """
    Heston stochastic volatility process

    Parameters
    ----------
    mu : Union[float, np.ndarray], the expected rate of return of the underlying asset
    theta : Union[float, np.ndarray], the long-term expected volatility level
    kappa : Union[float, np.ndarray], the mean reversion speed of the variance
    sigma : Union[float, np.ndarray], the volatility-of-variance
    rho : np.ndarray, the correlation coefficient between the two stochastic processes

    Examples
    --------
    >>> hp = HestonProcess(mu=0.05, kappa=0.8, sigma=0.001, theta=0.05, rho=-0.5)
    >>> hp.drift(x0=[10., 0.04])
    array([0.5  , 0.008])
    >>> hp.step(x0=[10., 0.04], dt=0.5, random_state=42)
    array([10.95245989,  0.04394794])
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, mu: float, theta: float, kappa: float, sigma: float, rho: float
    ) -> None:
        super().__init__()
        self.mu = mu
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho

    @property
    def correlation(self) -> np.ndarray:
        """Returns the correlation matrix of the processes"""
        return np.array([[1.0, self.rho], [self.rho, 1.0]])

    def coefs(self) -> Dict[str, float]:
        return dict(
            mu=self.mu,
            theta=self.theta,
            kappa=self.kappa,
            sigma=self.sigma,
            rho=self.rho,
        )

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        return x0 + dx

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        # floor volatility at zero before continuing (truncation method)
        vol = max(0.0, x0[1] ** 0.5)

        out = np.zeros_like(x0)
        out[0] = self.mu * x0[0]
        out[1] = self.kappa * (self.theta - vol * vol)
        return out

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # floor volatility near zero, but positive, to keep correlation effect
        vol = max(1e-6, x0[1] ** 0.5)

        # diffusion is the covariance diagonal times the cholesky correlation matrix
        cholesky = np.linalg.cholesky(self.correlation)
        volatility = np.diag([x0[0] * vol, self.sigma * vol])
        return volatility @ cholesky


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
