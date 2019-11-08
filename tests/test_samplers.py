"""Test sampling algorithms"""
import unittest

from pyesg.models.samplers import recursive_sampler


class TestRecursiveSampler(unittest.TestCase):
    """Test recursive sampler"""

    @staticmethod
    def simple_process(a, b, c, init, dW):
        """Create a simple process to test the sampler"""
        return a * (b - init) + c * dW

    def test_sampler_output_size(self):
        """Ensure the correct output size of the sampling algorithm"""
        n_scen, n_years, step_size = 1000, 30, 12
        samples = recursive_sampler(
            process=self.simple_process,
            params=dict(a=1, b=2, c=3),
            n_scen=n_scen,
            n_years=n_years,
            step_size=step_size,
            init=0.03,
            random_state=None,
        )
        self.assertEqual(samples.shape, (n_scen, 1 + n_years * step_size))

    def test_sampler_first_column_init(self):
        """Ensure the first value of every scenario is equal to init"""
        n_scen, init = 1000, 0.03
        samples = recursive_sampler(
            process=self.simple_process,
            params=dict(a=1, b=2, c=3),
            n_scen=1000,
            n_years=30,
            step_size=12,
            init=init,
            random_state=None,
        )
        self.assertListEqual(list(samples[:, 0]), [init] * n_scen)
