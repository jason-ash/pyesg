# pyESG
Economic Scenario Generator for Python.

## Objectives
I'm not aware of any comprehensive ESG library available for Python today. I think such a library should have the following components:

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
Open Source and licensed under MIT, Copyright (c) 2019 Jason Ash

## Examples

#### Vasicek Interest Rate Model
The project is in its early stages, but I've recently implemented the Vasicek interest rate model. At the moment, the API looks like this.

```python
import pyesg

# load a dataset of historical US Treasury rates, contained in pyesg.datasets
# ust is a pandas dataframe of rates for various maturities, indexed by year and month
ust = pyesg.datasets.load_ust_historical()

# for this example, we'll train on the following data:
# y - the 3-month US treasury rate
# X - the time value of each observation, starting at zero, increasing by monthly steps
y = ust.loc['3-month'].values
X = np.full(len(y), 1/12).cumsum()

# create a vasicek model object, just like you would create an estimator model from sklearn
vasicek = pyesg.Vasicek()

# fit the model by passing the X and y vectors; the model is now trained
vasicek.fit(X, y)

# sample future paths from the model
# specify the number of scenarios (e.g. 1000)
# length of projection (e.g. 30 years)
# and time step (e.g. weekly, or 52 time-steps per year)
scenarios = vasicek.sample(size=(1000, 30, 52))
```

#### Nelson-Siegel and Nelson-Siegel-Svensson Curve Interpolators
We can almost always observe interest rates at key maturities, for example, bonds trading with maturies of 1, 2, 3, 5, 7, or 10 years. If we want to estimate the interest rate for an 8-year bond, we need to interpolate between the observed values. Simple techniques like linear interpolation are possible, but have certain obvious disadvantages - namely that the interest rate curve is non-linear. Instead, better techniques like the Nelson-Siegel and Nelson-Siegel-Svensson interpolators might give better results. Both interpolators are availabe in `pyesg`.

```python
import pyesg
import matplotlib.pyplot as plt


# load a dataset of historical US Treasury rates, contained in pyesg.datasets
# ust is a pandas dataframe of rates for various maturities, indexed by year and month
ust = pyesg.datasets.load_ust_historical()

# we will be interpolating rates from the file:
# y - the observed US Treasury rate for the given maturity for a select observation date
# X - the maturity of the bond measured in years
y = data.iloc[-10].values
X = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

# create Nelson-Siegel and Nelson-Siegel-Svensson interpolator objects
# then fit both models using the historical data
ns = pyesg.NelsonSiegel()
nss = pyesg.NelsonSiegelSvensson()
ns.fit(X, y)
nss.fit(X, y)
```
