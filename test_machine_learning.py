from unittest import TestCase
from machine_learning import *
import numpy as np


class TestMachineLearning(TestCase):
    def test_train_linear_regression(self):
        # Without bias
        X = np.array([[-1.0], [1.0]])
        y = np.array([-1.5, 1.5])
        linear_regression = LinearRegression(X, y, False)
        np.testing.assert_almost_equal(linear_regression.predict(X), y)
        self.assertAlmostEqual(linear_regression.error(X, y), 0.0)

        # With bias
        X = np.array([[-1.0], [1.0]])
        y = np.array([0.5, 2.5])
        linear_regression = LinearRegression(X, y, True)
        np.testing.assert_almost_equal(linear_regression.predict(X), y)
        self.assertAlmostEqual(linear_regression.error(X, y), 0.0)

        # With polynomial
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 4.0, 9.0])
        linear_regression = LinearRegression(X, y, True, 2)
        np.testing.assert_almost_equal(linear_regression.predict(X), y)
        self.assertAlmostEqual(linear_regression.error(X, y), 0.0)

        # Overfitting
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 4.0, 9.0])
        X_test = np.array([[0.0], [1.5], [4.0]])
        y_test = np.array([0.0, 2.25, 16.0])
        linear_regression = LinearRegression(X_train, y_train, False, 9)
        self.assertAlmostEqual(linear_regression.error(X_train, y_train), 0.0)
        self.assertGreater(linear_regression.error(X_test, y_test), 0.0)

        # Underfitting
        linear_regression = LinearRegression(X_train, y_train, False, 1)
        self.assertGreater(linear_regression.error(X_train, y_train), 0.0)
        self.assertGreater(linear_regression.error(X_test, y_test), 0.0)

        # Best capacity
        linear_regression = LinearRegression(X_train, y_train, False, 2)
        self.assertAlmostEqual(linear_regression.error(X_train, y_train), 0.0)
        self.assertAlmostEqual(linear_regression.error(X_test, y_test), 0.0)

        # Regularization
        linear_regression = LinearRegression(X_train, y_train, False, 4, 1.0)
        self.assertLess(linear_regression.error(X_train, y_train), 0.01)
        self.assertLess(linear_regression.error(X_test, y_test), 1.0)

    def test_hyper_linear_regression(self):
        # Tuning hyper parameter with test data
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 4.0, 9.0])
        X_test = np.array([[0.0], [1.5], [4.0]])
        y_test = np.array([0.0, 2.25, 16.0])
        hyper_linear_regression = HyperLinearRegression(X_train, y_train)
        self.assertAlmostEqual(hyper_linear_regression.error(X_train, y_train), 0.0)
        self.assertEqual(hyper_linear_regression.model.degree, 2)
        self.assertAlmostEqual(hyper_linear_regression.error(X_test, y_test), 0.0)

    def test_nearest_neighbor(self):
        # Solving regression problem with nearest neighbor approach
        X = np.array([[-1.0], [1.0]])
        y = np.array([-1.5, 1.5])
        nearest_neighbor = NearestNeighbor(X, y)
        np.testing.assert_almost_equal(nearest_neighbor.predict(X), y)
        self.assertAlmostEqual(nearest_neighbor.error(X, y), 0.0)

        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 4.0, 9.0])
        X_test = np.array([[0.0], [1.5], [4.0]])
        y_test = np.array([0.0, 2.25, 16.0])
        nearest_neighbor = NearestNeighbor(X_train, y_train)
        np.testing.assert_almost_equal(nearest_neighbor.predict(X_train), y_train)
        self.assertAlmostEqual(nearest_neighbor.error(X_train, y_train), 0.0)
        self.assertGreater(nearest_neighbor.error(X_test, y_test), 0.0)

    def test_logistic_regression(self):
        # Binary classification with logistic regression
        X = np.array([[1.0, 1.0],
                      [-1.0, -1.0]])
        y = np.array([1, 0])

        model = LogisticRegression(X, y)
        prediction = model.predict(X)
        np.testing.assert_almost_equal(prediction, y)

    def test_support_vector_machine(self):
        # Learning XOR with gaussian kernel SVM
        X = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [1.0, 1.0],
                      [0.0, 1.0]])
        y = np.array([1, -1, 1, -1])

        model = SupportVectorMachine(X, y)
        prediction = model.predict(X)
        np.testing.assert_almost_equal(prediction, y, decimal=4)
