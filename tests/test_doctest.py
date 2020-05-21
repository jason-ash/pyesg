"""Explicitly add doctests to the unittest test suite"""
import doctest

from pyesg import utils
from pyesg.processes import (
    academy_rate_process,
    black_scholes_process,
    cox_ingersoll_ross_process,
    geometric_brownian_motion,
    heston_process,
    ornstein_uhlenbeck_process,
    wiener_process,
)


MODULES = [
    utils,
    academy_rate_process,
    black_scholes_process,
    cox_ingersoll_ross_process,
    geometric_brownian_motion,
    heston_process,
    ornstein_uhlenbeck_process,
    wiener_process,
]


def load_tests(loader, tests, ignored):  # pylint: disable=unused-argument
    """
    This function allows unittest to discover doctests in the module.

    It appears not to use the arguments (or do anything, really), but this is
    used internally by unittest to "discover" the tests it needs to run.

    References
    ----------
    https://stackoverflow.com/questions/5681330/using-doctests-from-within-unittests
    https://docs.python.org/2/library/unittest.html#load-tests-protocol
    """
    for module in MODULES:
        tests.addTests(doctest.DocTestSuite(module))
    return tests
