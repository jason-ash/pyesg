"""Estimator classes for interest rate curve interpolation"""
import numpy as np

from scipy.optimize import least_squares


class Interpolator:
    """Base class for Interpolators"""


class NelsonSiegel(Interpolator):
    """
    Nelson-Siegel Curve Interpolator

    Parameters
    ----------
    tau : float, optional, if provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process.
    """

    def __init__(self, tau=None):
        self._fit_tau = tau is None
        self.tau = tau

    def __repr__(self):
        return "Nelson-Siegel Estimator"

    @classmethod
    def formula(cls, t, beta0, beta1, beta2, tau):
        """Returns the Nelson-Siegel estimate of a rate at maturity t"""
        return (
            beta0
            + beta1 * (1 - np.exp(-t * tau)) / (t * tau)
            + beta2 * ((1 - np.exp(-t * tau)) / (t * tau) - np.exp(-t * tau))
        )

    def fit(self, X, y):
        """
        Fits the Nelson-Siegel estimator using ordinary least squares

        Parameters
        ----------
        X : np.array of maturies, must be >0
        y : np.array of rates corresponding to each maturity

        Returns
        -------
        self : returns an instance of self
        """
        if self._fit_tau:
            # solve for all betas and tau
            def f(params, t, y):
                return self.formula(t, *params) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 1.0], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.tau = ls.x
        else:
            # keep tau fixed; solve for all betas
            def f(params, t, y):
                return self.formula(t, *params, tau=self.tau) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2 = ls.x
        return self

    def predict(self, X):
        """Returns the predicted values from an array of independent values"""
        if not hasattr(self, "beta0"):
            raise RuntimeError("Must call 'fit' first!")
        return self.formula(X, self.beta0, self.beta1, self.beta2, self.tau)


class NelsonSiegelSvensson(Interpolator):
    """
    Nelson-Siegel-Svensson Curve Interpolator

    Parameters
    ----------
    tau0 : float, optional, if both tau0 and tau1 are provided, then neither value is
        solved when 'fit' is called; otherwise, if neither or just one value is provided,
        then both values will be provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process
    tau1 : float, optional, if both tau0 and tau1 are provided, then neither value is
        solved when 'fit' is called; otherwise, if neither or just one value is provided,
        then both values will be provided, then this value is not solved when 'fit' is called
        otherwise, it is considered a free variable as part of the fitting process
    """

    def __init__(self, tau0=None, tau1=None):
        self._fit_tau = (tau0 is None) or (tau1 is None)
        self.tau0 = tau0
        self.tau1 = tau1

    def __repr__(self):
        return "Nelson-Siegel-Svensson Estimator"

    @classmethod
    def formula(cls, t, beta0, beta1, beta2, beta3, tau0, tau1):
        """Returns the Nelson-Siegel-Svensson estimate of a rate at maturity t"""
        return (
            beta0
            + beta1 * (1 - np.exp(-t * tau0)) / (t * tau0)
            + beta2 * ((1 - np.exp(-t * tau0)) / (t * tau0) - np.exp(-t * tau0))
            + beta3 * ((1 - np.exp(-t * tau1)) / (t * tau1) - np.exp(-t * tau1))
        )

    def fit(self, X, y):
        """
        Fits the Nelson-Siegel-Svensson estimator using ordinary least squares

        Parameters
        ----------
        X : np.array of maturies, must be >0
        y : np.array of rates corresponding to each maturity

        Returns
        -------
        self : returns an instance of self
        """
        if self._fit_tau:
            # solve for all betas and taus
            def f(params, t, y):
                return self.formula(t, *params) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 0.01, 1.0, 1.0], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau0, self.tau1 = ls.x
        else:
            # keep taus fixed; solve for all betas
            def f(params, t, y):
                return self.formula(t, *params, tau0=self.tau0, tau1=self.tau1) - y

            ls = least_squares(f, x0=[0.01, 0.01, 0.01, 0.01], args=(X, y))
            self.beta0, self.beta1, self.beta2, self.beta3 = ls.x
        return self

    def predict(self, X):
        """Returns the predicted values from an array of independent values"""
        if not hasattr(self, "beta0"):
            raise RuntimeError("Must call 'fit' first!")
        return self.formula(
            X, self.beta0, self.beta1, self.beta2, self.beta3, self.tau0, self.tau1
        )
