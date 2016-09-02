from unittest import TestCase
from optimization import *
import numpy as np


def cost_scalar(x):
    return x * x


def derivative(x):
    return 2.0 * x


def second_derivative(x):
    return 2.0


def cost_vector(x):
    return np.sum(x * x)


def gradient(x):
    return 2.0 * x


def hessian(x):
    return 2.0 * np.identity(x.size)


class TestOptimization(TestCase):
    def test_gradient_descent(self):
        # Scalar test
        i, x, cost = run_iterations(gradient_descent(cost_scalar, derivative, 10.0, 0.1), 1000)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(cost, 0.0)
        self.assertGreater(i, 1)
        self.assertLess(i, 200)

        # Vector test
        i, x, cost = run_iterations(gradient_descent(cost_vector, gradient, np.array([10.0, 0.0]), 0.1), 1000)
        np.testing.assert_almost_equal(x, np.zeros(2))
        self.assertAlmostEqual(cost, 0.0)
        self.assertGreater(i, 1)
        self.assertLess(i, 200)

    def test_newtons_method(self):
        # Scalar test
        i, x, cost = run_iterations(newtons_method(cost_scalar, derivative, second_derivative, 10.0, 1.0), 1000)
        self.assertEqual(i, 1)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(cost, 0.0)

        # Vector test
        i, x, cost = run_iterations(newtons_method(cost_vector, gradient, hessian, np.array([10.0, 0.0]), 1.0), 1000)
        self.assertEqual(i, 1)
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertAlmostEqual(cost, 0.0)
