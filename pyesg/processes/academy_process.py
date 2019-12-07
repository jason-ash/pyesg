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
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        beta1: float = 0.00509,  # reversion strength for long-rate process
        beta2: float = 0.02685,  # reversion strength for spread process
        beta3: float = 0.04001,  # reversion strength for volatility process
        rho12: float = -0.19197,  # correlation between long-rate & spread
        rho13: float = 0.0,  # correlation between long-rate & volatility
        rho23: float = 0.0,  # correlation between spread & volatility
        tau1: float = 0.035,  # mean reversion value for long-rate process
        tau2: float = 0.01,  # mean reversion value for spread process
        tau3: float = 0.0287,  # mean reversion value for volatility process
        theta: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.rho12 = rho12
        self.rho13 = rho13
        self.rho23 = rho23
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.theta = theta
        self.correlation = np.array(
            [[1.0, rho12, rho13], [rho12, 1.0, rho23], [rho13, rho23, 1.0]]
        )

    def coefs(self) -> Dict[str, float]:
        """Returns a dictionary of the process coefficients"""
        raise NotImplementedError()

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        """Returns the drift component of the stochastic process"""
        raise NotImplementedError()

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        """Returns the diffusion component of the stochastic process"""
        raise NotImplementedError()
