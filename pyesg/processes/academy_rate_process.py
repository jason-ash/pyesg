"""
American Academy of Actuaries stochastic log volatility interest rate process

References
----------
https://www.actuary.org/sites/default/files/pdf/life/lbrc_dec08.pdf, page 8
"""
from typing import Dict
import numpy as np

from pyesg.stochastic_process import StochasticProcess


class AcademyRateProcess(StochasticProcess):
    """
    American Academy of Actuaries stochastic log volatility process. Models three linked
    processes:
        1. long-term-rate (internally modeled as a process on the log-rate)
        2. nominal spread between long-term rate and short-term rate
        3. monthly-volatility of the long-rate process (internally modeled as log-vol)

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
    >>> arp = AcademyRateProcess.example()
    >>> arp.correlation
    array([[ 1.     , -0.19197,  0.     ],
           [-0.19197,  1.     ,  0.     ],
           [ 0.     ,  0.     ,  1.     ]])

    >>> arp.drift(x0=[0.03, 0.0024, 0.03])
    array([ 0.03236509,  0.00207876, -0.02126944])

    >>> arp.diffusion(x0=[0.03, 0.0024, 0.03])
    array([[ 0.10392305,  0.        ,  0.        ],
           [-0.00082753,  0.00423055,  0.        ],
           [ 0.        ,  0.        ,  0.39799063]])

    >>> arp.expectation(x0=[0.03, 0.0024, 0.03], dt=1./12)
    array([0.03008102, 0.00257323, 0.02994687])

    >>> arp.standard_deviation(x0=[0.03, 0.0024, 0.03], dt=1./12)
    array([[ 0.03      ,  0.        ,  0.        ],
           [-0.00023889,  0.00122126,  0.        ],
           [ 0.        ,  0.        ,  0.11489   ]])

    >>> arp.step(x0=[0.03, 0.0024, 0.03], dt=1./12, random_state=42)
    array([0.03053263, 0.00228572, 0.03226032])

    References
    ----------
    https://www.actuary.org/sites/default/files/pdf/life/lbrc_dec08.pdf, page 8
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
        super().__init__(dim=3)
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

    def _apply(self, x0: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # long-rate (x0[0]) is modeled internally as a log process, so we use exp
        # spread (x0[1]) is an arithmetic process
        # volatility (x0[2]) is modeled internally as a log-process, so we use exp
        out = np.empty_like(x0)

        if x0.ndim == 1:
            longrate_ = np.s_[0]
            spread_ = np.s_[1]
            volatility_ = np.s_[2]
        else:
            longrate_ = np.s_[:, 0]
            spread_ = np.s_[:, 1]
            volatility_ = np.s_[:, 2]

        out[longrate_] = x0[longrate_] * np.exp(dx[longrate_])
        out[spread_] = x0[spread_] + dx[spread_]
        out[volatility_] = x0[volatility_] * np.exp(dx[volatility_])
        return out

    def _drift(self, x0: np.ndarray) -> np.ndarray:
        # x0 is an array of [long-rate, nominal spread, volatility]
        # create a new array to store the output, then simultaneously update all terms
        drift = np.empty_like(x0)

        # how do we need to slice the input array? store these before proceeding
        if x0.ndim == 1:
            longrate_ = np.s_[0]
            spread_ = np.s_[1]
            volatility_ = np.s_[2]
        else:
            longrate_ = np.s_[:, 0]
            spread_ = np.s_[:, 1]
            volatility_ = np.s_[:, 2]

        # --volatility of the long-rate process--
        # internally we model the monthly-log-volatility of the long-rate process; this
        # variable follows an ornstein-uhlenbeck process with mean reversion parameter
        # tau3, and mean reversion speed beta3. Because we are modeling log-volatiliy,
        # we take the log of tau3 and the initial volatility level, x0[2]. This class's
        # "apply" method will use exponentiation to update the value of volatility,
        # so the output of the model is "converted" back to non-log volatility.
        drift[volatility_] = self.beta3 * np.log(self.tau3 / x0[volatility_])

        # --spread between long-rate and short-rate--
        # the spread follows an ornstein-uhlenbeck process with mean reversion parameter
        # tau2 and mean reversion speed beta2. We model nominal spread, so the effect is
        # additive to the original level of the spread. We also add a component based on
        # the level of the log-long-rate compared to its mean reversion rate, tau1. This
        # is multiplied by a factor, phi, and added to the drift component of the spread
        drift[spread_] = self.beta2 * (self.tau2 - x0[spread_]) + self.phi * np.log(
            x0[longrate_] / self.tau1
        )

        # --long-term interest rate--
        # internally we model the log-long-term rate as an ornstein-uhlenbeck process
        # with mean reversion level tau1 and mean reversion speed beta1. We also add an
        # effect based on the level of the spread relative to its long term mean times a
        # factor, psi. Before we calculate the drift we set upper and lower bounds on
        # the long term rate so it falls within a certain range. This range may be
        # exceeded based on the random perturbations that are added later, which is ok.
        # Similar to volatility, we apply the changes here using exponentiation.
        drift[longrate_] = self.beta1 * np.log(self.tau1 / x0[longrate_]) + self.psi * (
            self.tau2 - x0[spread_]
        )
        drift[longrate_] = np.minimum(
            np.log(self.long_rate_max / x0[longrate_]), drift[longrate_]
        )
        drift[longrate_] = np.maximum(
            np.log(self.long_rate_min / x0[longrate_]), drift[longrate_]
        )
        return drift

    def _diffusion(self, x0: np.ndarray) -> np.ndarray:
        # diffusion is the covariance diagonal times the cholesky correlation matrix
        # x0 is an array of [long-rate, spread, volatility]

        # how do we need to slice the input array? store these before proceeding
        if x0.ndim == 1:
            longrate_ = np.s_[0]
            spread_ = np.s_[1]
            volatility_ = np.s_[2]
        else:
            longrate_ = np.s_[:, 0]
            spread_ = np.s_[:, 1]
            volatility_ = np.s_[:, 2]

        # diffusion matrix starts as a copy of the initial array; we'll update below
        diffusion = np.empty_like(x0)

        # --volatility of the long-rate process--
        # this is just the value of sigma3 - the volatility of the volatility parameter
        diffusion[volatility_] = self.sigma3

        # --spread between long-rate and short-rate--
        # this volatility follows a generalized ornstein-uhlenbeck process with an extra
        # factor of multiplying by the long rate raised to a power of theta
        diffusion[spread_] = self.sigma2 * x0[longrate_] ** self.theta

        # --long-term interest rate--
        # here we use the volatility value calculated by the third stochastic process.
        # however, because that volatility process models *monthly* volatility, we need
        # to scale it to be *annual* volatility, so we multiply by sqrt(12)
        diffusion[longrate_] = x0[volatility_] * 12 ** 0.5

        # finally, we output the matrix product of volatility (as a diagonal matrix)
        # with the cholesky decomposition of the correlation matrix.
        cholesky = np.linalg.cholesky(self.correlation)

        if x0.ndim == 1:
            # creates a (3, 3) array with a single diffusion matrix
            diffusion = np.diag(diffusion)
        else:
            # creates a (N, 3, 3) array with a diffusion matrix per sample in x0
            diffusion = np.eye(diffusion.shape[1]) * diffusion[:, None]
        return diffusion @ cholesky

    @classmethod
    def example(cls):
        return cls(
            beta1=0.00509,
            beta2=0.02685,
            beta3=0.04001,
            rho12=-0.19197,
            rho13=0.0,
            rho23=0.0,
            sigma2=0.04148,
            sigma3=0.11489,
            tau1=0.035,
            tau2=0.01,
            tau3=0.0287,
            theta=1.0,
            phi=0.0002,
            psi=0.25164,
            long_rate_max=0.18,
            long_rate_min=0.0115,
        )
