from unittest import TestCase
from linear_regression import *
import numpy as np


class TestLinearRegression(TestCase):
    def test_train_linear_regression(self):
        # Without bias
        X = np.array([[-1.0], [1.0]])
        y = np.array([-1.5, 1.5])
        linear_regression = LinearRegression(X, y, False)
        np.testing.assert_almost_equal(linear_regression.predict(X), y)
        self.assertAlmostEqual(linear_regression.error(X, y), 0.0)

        X = np.array([[-1.0], [1.0]])
        y = np.array([0.5, 2.5])
        linear_regression = LinearRegression(X, y, False)
        self.assertGreater(linear_regression.error(X, y), 0.0)

        # With bias
        X = np.array([[-1.0], [1.0]])
        y = np.array([0.5, 2.5])
        linear_regression = LinearRegression(X, y, True)
        np.testing.assert_almost_equal(linear_regression.predict(X), y)
        self.assertAlmostEqual(linear_regression.error(X, y), 0.0)
