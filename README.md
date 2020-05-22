# pyesg
Economic Scenario Generator for Python.

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyesg.svg)](https://pypi.python.org/pypi/pyesg/)
[![PyPI version](https://badge.fury.io/py/pyesg.svg)](https://badge.fury.io/py/pyesg)
[![Coverage Status](https://coveralls.io/repos/github/jason-ash/pyesg/badge.svg?branch=master)](https://coveralls.io/github/jason-ash/pyesg?branch=master)

## Objectives
I think an economic scenario generator library should have the following components:

1. A suite of stochastic models - inspired by the "fit"/"predict" scikit-learn API
    - that are easy to calibrate ("fit") using historical data, or to manually provide parameters.
    - that can simulate future economic paths ("predict")
2. A suite of model and scenario evaluation tools to
    - evaluate the goodness of fit of models
    - calculate significance measures of generated scenarios (to select subsets of scenarios if desired)
3. Replicate the existing SOA and AAA Excel generator so actuarial teams can migrate to Python.
4. Minimal dependencies - relying on the standard scientific python stack: `numpy`, `pandas`, and `scipy`, with optional plotting from `matplotlib`.

I expect that these objectives may shift or expand as I continue working on the library. Please let me know if you think anything is missing!

## Installation
You can install pyesg from the command prompt with the following:

```
pip install pyesg
```

## License
Open Source and licensed under MIT, Copyright &copy; 2019 Jason Ash

## Examples

#### Stochastic Processes
The project is in its early stages, but I've recently implemented several stochastic processes. Currently the stochastic processes generate single scenarios at a time, but I'm working on a faster batch sampling method. At the moment, the API looks like this.

```python
# pyesg currently has stochastic processes for:
#    - OrnsteinUhlenbeckProcess (Vasicek model)
#    - CoxIngersollRossProcess (square root model)
#    - AcademyRateProcess (American Academy of Actuaries interest rate model)
#    - Heston Process (stochastic volatility process)
#    - Wiener Process (simple random walk)
#    - BlackScholesProcess (geometric brownian motion)
from pyesg import OrnsteinUhlenbeckProcess

# create a Vasicek Model (Ornstein Uhlenbeck Process) with given paramters
oup = OrnsteinUhlenbeckProcess(mu=0.05, sigma=0.015, theta=0.15)

# generate a single interest rate scenario by specifying the time step (steps/yr),
# the number of steps to project, and the starting value, x0. Below we simulate
# weekly timesteps for one year starting at an initial value of 3.0%.
oup.scenario(x0=0.03, dt=1/52, n_step=52, random_state=42)
# array([0.03      , 0.03109092, 0.03085786, 0.03226035, 0.03547962,
#        0.03503443, 0.03459057, 0.03791998, 0.03955119, 0.03860476,
#        0.03976623, 0.03883178, 0.03789522, 0.03843345, 0.03448695,
#        0.03094365, 0.02982899, 0.02778036, 0.02849813, 0.02667135,
#        0.02380088, 0.02692519, 0.02652211, 0.0267303 , 0.02383377,
#        0.02277686, 0.02308612, 0.02076955, 0.02163536, 0.02046778,
#        0.01994621, 0.01878128, 0.02272431, 0.02277491, 0.02065327,
#        0.02244892, 0.01998889, 0.02050992, 0.01651863, 0.01385242,
#        0.01436618, 0.01600508, 0.01645961, 0.01631579, 0.01578663,
#        0.01280981, 0.01141972, 0.01057282, 0.0128855 , 0.01370733,
#        0.01014468, 0.01093378, 0.01024545])
```

#### Nelson-Siegel and Nelson-Siegel-Svensson Curve Interpolators
We can almost always observe interest rates at key maturities, for example, bonds trading with maturies of 1, 2, 3, 5, 7, or 10 years. If we want to estimate the interest rate for an 8-year bond, we need to interpolate between the observed values. Simple techniques like linear interpolation are possible, but have certain obvious disadvantages - namely that the interest rate curve is non-linear. Instead, better techniques like the Nelson-Siegel and Nelson-Siegel-Svensson interpolators might give better results. Both interpolators are availabe in `pyesg`.

```python
import numpy as np
from pyesg import NelsonSiegelInterpolator, SvenssonInterpolator
from pyesg.datasets import load_ust_historical

# load a dataset of historical US Treasury rates, contained in pyesg.datasets
# ust is a pandas dataframe of rates for various maturities, indexed by year and month
ust = load_ust_historical()

# we will be interpolating rates from the file:
# y - the observed US Treasury rate for the given maturity for a select observation date
# X - the maturity of the bond measured in years
y = ust.iloc[-10].values
X = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

# create Nelson-Siegel and Nelson-Siegel-Svensson interpolator objects
# then fit both models using the historical data
nelson_siegel = pyesg.NelsonSiegelInterpolator()
svensson = pyesg.SvenssonInterpolator()
nelson_siegel.fit(X, y)
svensson.fit(X, y)

# predict values for each maturity from 1 to 30 years
nelson_siegel.predict(np.arange(1, 31, 1))
# array([0.02033871, 0.02252733, 0.02403659, 0.02510373, 0.02587762,
#        0.02645304, 0.02689131, 0.02723275, 0.02750438, 0.02772458,
#        0.02790617, 0.02805818, 0.02818715, 0.02829786, 0.0283939 ,
#        0.02847798, 0.02855218, 0.02861815, 0.02867718, 0.02873031,
#        0.02877839, 0.02882209, 0.02886199, 0.02889857, 0.02893222,
#        0.02896329, 0.02899205, 0.02901876, 0.02904362, 0.02906683])
```

<img src="docs/images/NelsonSiegel.png" width="600">
