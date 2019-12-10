"""
American Academy of Actuaries stochastic log volatility interest rate process

References
----------
https://www.actuary.org/sites/default/files/pdf/life/lbrc_dec08.pdf, page 8
"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import JointStochasticProcess


class AcademyRateProcess(JointStochasticProcess):
    """
    American Academy of Actuaries stochastic log volatility process. Models three linked
    processes:
        1 : long-term-rate (internally modeled as a process on the log-rate)
        2 : nominal spread between long-term rate and short-term rate
        3 : monthly-volatility of the long-rate process (internally modeled as log-vol)

    NOTE : most parameters provided as defaults are _monthly_ parameters, not _annual_
        parameters; to keep consistent with the Academy Excel workbook, these values are
        kept as the monthly defaults. Internally, the model converts them to annual
        values. The Excel workbook is scaled to monthly timesteps, whereas this model is
        scaled to annual timesteps. We can replicate the Excel results here by calling
        `dt=1./12` to get monthly output steps.

    Parameters
    ----------
    beta1 : float, default 0.00509, reversion strength for long-rate process
    beta2 : float, default 0.02685, reversion strength for spread process
    beta3 : float, default 0.04001, reversion strength for volatility process
    rho12 : float, default -0.19197, correlation between long-rate & spread
    rho13 : float, default 0.0, correlation between long-rate & volatility
    rho23 : float, default 0.0, correlation between spread & volatility
    sigma2 : float, default 0.04148, volatility of the spread process
    sigma3 : float, default 0.11489, volatility of the volatility process
    tau1 : float, default 0.035, mean reversion value for long-rate process
    tau2 : float, default 0.01, mean reversion value for spread process
    tau3 : float, default 0.0287, mean reversion value for volatility process
    theta : float, default 1.0, spread volatility factor exponent
    phi : float, default 0.0002, spread tilting parameter
    psi : float, default 0.25164, steepness adjustment
    long_rate_max : float, default 0.18, soft cap of the long rate before perturbing
    long_rate_min : float, default 0.0115, soft floor of the long rate before perturbing

    Examples
    --------
    >>> arp = AcademyRateProcess()
    >>> arp.correlation
    array([[ 1.     , -0.19197,  0.     ],
           [-0.19197,  1.     ,  0.     ],
           [ 0.     ,  0.     ,  1.     ]])
    >>> arp.drift(x0=[0.0287, 0.0024, 0.03])
    array([ 0.03507095,  0.00197244, -0.02126944])
    >>> arp.diffusion(x0=[0.0287, 0.0024, 0.03])
    array([[ 0.10392305,  0.        ,  0.        ],
           [-0.00079167,  0.00404723,  0.        ],
           [ 0.        ,  0.        ,  0.39799063]])
    >>> arp.expectation(x0=[0.0287, 0.0024, 0.03], dt=1./12)
    array([0.028784  , 0.00256437, 0.02994687])
    >>> arp.standard_deviation(x0=[0.0287, 0.0024, 0.03], dt=1./12)
    array([[ 0.03      ,  0.        ,  0.        ],
           [-0.00022854,  0.00116833,  0.        ],
           [ 0.        ,  0.        ,  0.11489   ]])
    >>> arp.step(x0=[0.0287, 0.0024, 0.03], dt=1./12, random_state=42)
    array([0.02921614, 0.00228931, 0.03226032])
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        beta1: float = 0.00509,  # reversion strength for long-rate process
        beta2: float = 0.02685,  # reversion strength for spread process
        beta3: float = 0.04001,  # reversion strength for volatility process
        rho12: float = -0.19197,  # correlation between long-rate & spread
        rho13: float = 0.0,  # correlation between long-rate & volatility
        rho23: float = 0.0,  # correlation between spread & volatility
        sigma2: float = 0.04148,  # volatility of the spread process
        sigma3: float = 0.11489,  # volatility of the volatility process
        tau1: float = 0.035,  # mean reversion value for long-rate process
        tau2: float = 0.01,  # mean reversion value for spread process
        tau3: float = 0.0287,  # mean reversion value for volatility process
        theta: float = 1.0,  # spread volatility factor exponent
        phi: float = 0.0002,  # spread tilting parameter
        psi: float = 0.25164,  # steepness adjustment
        long_rate_max: float = 0.18,  # soft cap of the long rate before perturbing
        long_rate_min: float = 0.0115,  # soft floor of the long rate before perturbing
    ) -> None:
        super().__init__()
        self.beta1 = beta1 * 12  # annualize the monthly-based parameter
        self.beta2 = beta2 * 12  # annualize the monthly-based parameter
        self.beta3 = beta3 * 12  # annualize the monthly-based parameter
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23
        self.sigma2 = sigma2 * 12 ** 0.5  # annualize the monthly-based parameter
        self.sigma3 = sigma3 * 12 ** 0.5  # annualize the monthly-based parameter
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.theta = theta
        self.phi = phi * 12  # annualize the monthly-based parameter
        self.psi = psi * 12  # annualize the monthly-based parameter
        self.long_rate_max = long_rate_max
        self.long_rate_min = long_rate_min

    @property
    def correlation(self) -> np.ndarray:
        """Returns the correlation matrix of the processes"""
        return np.array(
            [
                [1.0, self.rho12, self.rho13],
                [self.rho12, 1.0, self.rho23],
                [self.rho13, self.rho23, 1.0],
            ]
        )

    def coefs(self) -> Dict[str, float]:
        return dict(
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            rho12=self.rho12,
            rho13=self.rho13,
            rho23=self.rho23,
            sigma2=self.sigma2,
            sigma3=self.sigma3,
            tau1=self.tau1,
            tau2=self.tau2,
            tau3=self.tau3,
            theta=self.theta,
            phi=self.phi,
            psi=self.psi,
            long_rate_max=self.long_rate_max,
            long_rate_min=self.long_rate_min,
        )

    def apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # arithmetic addition to update x0
        out = x0.copy()
        out[0] = x0[0] * np.exp(dx[0])
        out[1] = x0[1] + dx[1]
        out[2] = x0[2] * np.exp(dx[2])
        return out

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        # x0 is an array of [long-rate, nominal spread, volatility]
        # create a new array to store the output, then simultaneously update all terms
        out = x0.copy()

        # updating log-volatility
        out[2] = self.beta3 * np.log(self.tau3 / x0[2])

        # updating spread
        out[1] = self.beta2 * (self.tau2 - x0[1]) + self.phi * np.log(x0[0] / self.tau1)

        # expectation for the log-long-term rate
        out[0] = self.beta1 * np.log(self.tau1 / x0[0]) + self.psi * (self.tau2 - x0[1])
        out[0] = min(self.long_rate_max - x0[0], out[0])
        out[0] = max(self.long_rate_min - x0[0], out[0])
        return out

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion is the covariance diagonal times the cholesky correlation matrix
        # x0 is an array of [long-rate, spread, volatility]
        cholesky = np.linalg.cholesky(self.correlation)
        volatility = np.diag(
            [
                x0[2] * 12 ** 0.5,  # annualize the monthly-based parameter
                self.sigma2 * x0[0] ** self.theta,
                self.sigma3,
            ]
        )
        return volatility @ cholesky


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
