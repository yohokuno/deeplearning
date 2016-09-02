from unittest import TestCase
from optimization import *


def cost_function(x):
    return x * x


def derivative(x):
    return 2.0 * x


def second_derivative(x):
    return 2.0


class TestOptimization(TestCase):
    def test_gradient_descent_scalar(self):
        # argmin_x x * x = 0
        i, x, cost = run_iterations(gradient_descent_scalar(cost_function, derivative, 10.0, 0.1), 1000)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(cost, 0.0)
        self.assertGreater(i, 1)
        self.assertLess(i, 1000-1)

    def test_newtons_method_scalar(self):
        # Quadratic function always converges with 1 iteration of Newton's method with learning ratio 1
        i, x, cost = run_iterations(newtons_method_scalar(cost_function, derivative, second_derivative, 10.0, 1.0), 1000)
        self.assertEqual(i, 1)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(cost, 0.0)
