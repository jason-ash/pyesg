"""Test sampling algorithms"""
# import unittest

# from pyesg.models import recursive_sampler


# class TestRecursiveSampler(unittest.TestCase):
#     """Test recursive sampler"""

#     @staticmethod
#     def simple_process(x, dt, dW):
#         """Create a simple process to test the sampler"""
#         a, b, c = 0.15, 0.045, 0.015
#         return a * (b - x) * dt + c * dt ** 0.5 * dW

#     def test_sampler_output_size(self):
#         """Ensure the correct output size of the sampling algorithm"""
#         n_scen, n_years, step_size = 1000, 30, 12
#         samples = recursive_sampler(
#             process=self.simple_process,
#             n_scen=n_scen,
#             n_years=n_years,
#             step_size=step_size,
#             init=0.03,
#             random_state=None,
#         )
#         self.assertEqual(samples.shape, (n_scen, 1 + n_years * step_size))

#     def test_sampler_first_column_init(self):
#         """Ensure the first value of every scenario is equal to init"""
#         n_scen, init = 1000, 0.03
#         samples = recursive_sampler(
#             process=self.simple_process,
#             n_scen=1000,
#             n_years=30,
#             step_size=12,
#             init=init,
#             random_state=None,
#         )
#         self.assertListEqual(list(samples[:, 0]), [init] * n_scen)
