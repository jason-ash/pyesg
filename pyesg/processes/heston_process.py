"""Heston stochastic volatility process"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


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
        super().__init__(dim=2)
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
        # x0 is an array of [underlying asset, volatility]
        drift = np.empty_like(x0, dtype=np.float64)

        # how do we need to slice the input array? store these before proceeding
        if x0.ndim == 1:
            underlying_ = np.s_[0]
            volatility_ = np.s_[1]
        else:
            underlying_ = np.s_[:, 0]
            volatility_ = np.s_[:, 1]

        # floor volatility at zero before continuing (truncation method)
        volatility = np.maximum(0.0, x0[volatility_]) ** 0.5
        drift[volatility_] = self.kappa * (self.theta - volatility * volatility)
        drift[underlying_] = self.mu * x0[underlying_]
        return drift

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # how do we need to slice the input array? store these before proceeding
        if x0.ndim == 1:
            underlying_ = np.s_[0]
            volatility_ = np.s_[1]
        else:
            underlying_ = np.s_[:, 0]
            volatility_ = np.s_[:, 1]

        diffusion = np.empty_like(x0)
        # floor volatility close to zero, but not zero, in order to preserve correlation
        diffusion[volatility_] = np.maximum(0.000001, x0[volatility_]) ** 0.5
        diffusion[underlying_] = x0[underlying_] * diffusion[volatility_]
        diffusion[volatility_] = diffusion[volatility_] * self.sigma

        if x0.ndim == 1:
            # create a (2, 2) diffusion matrix
            diffusion = np.diag(diffusion)
        else:
            # create a (N, 2, 2) diffusion matrix for the number of samples in x0
            diffusion = np.eye(diffusion.shape[1]) * diffusion[:, None]

        # diffusion is the covariance diagonal times the cholesky correlation matrix
        cholesky = np.linalg.cholesky(self.correlation)
        return diffusion @ cholesky

    @classmethod
    def example(cls):
        return cls(mu=0.05, kappa=0.8, sigma=0.001, theta=0.05, rho=-0.5)
