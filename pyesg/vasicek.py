"""Vasicek stochastic short-rate model"""
import numpy as np


class Vasicek:
    """
    Vasicek stochastic short-rate model

    r(t) = r(t) + k*(theta - r(t))*dt + sigma*dt**0.5*norm(0,1)

    Parameters
    ----------
    k : float, the rate of mean reversion - fit by MLE
    theta : float, the long-term interest rate level - fit by MLE
    sigma : float, the volatility of rates - fit by MLE
    """

    def __init__(self) -> None:
        self.k: Optional[float] = None  # fit parameter
        self.theta: Optional[float] = None  # fit parameter
        self.sigma: Optional[float] = None  # fit parameter

    def __repr__(self) -> str:
        return "Vasicek Model"

    @classmethod
    def _log_likelihod(
        cls, k: float, theta: float, sigma: float, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Log likelihood function, given model parameters and observed data

        Reference
        ---------
        https://pdfs.semanticscholar.org/dc0c/75b5d0002607046277a6768b9b54f34877c3.pdf
        """
        B = (1 - np.exp(-k * (X[1:] - X[:-1]))) / k
        m = theta * k * B + y[:-1] * (1 - k * B)
        v = sigma ** 2 * (B - k / 2 * B ** 2)
        return (np.log(2 * np.pi * v) + (y[1:] - m) ** 2 / v).sum() / -2

    @property
    def _fitted_params(self) -> Dict:
        """
        Returns a dictionary of fitted model parameters.
        Parameters default to None if they haven't been fitted yet.
        """
        return dict(k=self.k, theta=self.theta, sigma=self.sigma)

    @property
    def _check_fitted(self) -> bool:
        """Returns a boolean indicating whether or not the model has been fitted"""
        return all(x is not None for x in self._fitted_params.values())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Vasicek":
        """
        Fits the parameters of the Vasicek model using maximum likelihood.

        Parameters
        ----------
        X : np.array of timesteps in years,
            e.g. array([ 0.08333333,  0.16666667,  0.25]
        y : np.array of short-rates at each timestep
            e.g. array([2.190e-02, 2.160e-02, 2.110e-02]
        """
        # minimize the negative log-likelihood
        f = lambda params, X, y: -1 * self._log_likelihood(*params, X=X, y=y)
        mle = minimize(f, x0=[0.5, 0.1, 0.1], args=(X, y), method="Nelder-Mead")
        if mle.success:
            self.k, self.theta, self.sigma = mle.x
            return self
        else:
            raise RuntimeError("Model failed to converge")
