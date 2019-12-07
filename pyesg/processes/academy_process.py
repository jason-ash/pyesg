"""
American Academy of Actuaries stochastic log volatility process

References
----------
https://www.actuary.org/sites/default/files/pdf/life/lbrc_dec08.pdf, page 8
"""
from typing import Dict
import numpy as np

from pyesg.processes import JointStochasticProcess


class AcademyRateProcess(JointStochasticProcess):
    """
    American Academy of Actuaries stochastic log volatility process

    Examples
    --------
    >>> arp = AcademyRateProcess()
    >>> arp.correlation
    array([[ 1.     , -0.19197,  0.     ],
           [-0.19197,  1.     ,  0.     ],
           [ 0.     ,  0.     ,  1.     ]])
    >>> arp.diffusion(x0=[0.0287, 0.0024, 0.0287])
    array([[ 0.0287    ,  0.        ,  0.        ],
           [-0.00022854,  0.00116833,  0.        ],
           [ 0.        ,  0.        ,  0.11489   ]])
    >>> arp.standard_deviation(x0=[0.0287, 0.0024, 0.0287], dt=0.25)
    array([[ 0.01435   ,  0.        ,  0.        ],
           [-0.00011427,  0.00058417,  0.        ],
           [ 0.        ,  0.        ,  0.057445  ]])
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
    ) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.theta = theta
        self.phi = phi
        self.psi = psi

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
        )

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion is the covariance diagonal times the cholesky correlation matrix
        # x0 is an array of [long-rate, spread, volatility]
        cholesky = np.linalg.cholesky(self.correlation)
        volatility = np.diag([x0[2], self.sigma2 * x0[0] ** self.theta, self.sigma3])
        return volatility @ cholesky


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
