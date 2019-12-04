"""Test pyesg process classes"""
import doctest

from pyesg import processes


# pylint: disable=unused-argument,line-too-long
def load_tests(loader, tests, ignored):
    """
    This function allows unittest to discover doctests in the module.

    It appears not to use the arguments (or do anything, really), but this is
    used internally by unittest to "discover" the tests it needs to run.

    References
    ----------
    https://stackoverflow.com/questions/5681330/using-doctests-from-within-unittests
    https://docs.python.org/2/library/unittest.html#load-tests-protocol
    """
    tests.addTests(doctest.DocTestSuite(processes))
    return tests
