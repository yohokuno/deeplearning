from unittest import TestCase
from optimization import *


class TestOptimization(TestCase):
    def test_gradient_descent_scalar(self):
        cost_function = lambda x: x * x
        derivative_function = lambda x: 2 * x
        iterator = gradient_descent_scalar(cost_function, derivative_function, 1.0, 100, 0.1)
        for x, cost in iterator:
            pass
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(cost, 0.0)
