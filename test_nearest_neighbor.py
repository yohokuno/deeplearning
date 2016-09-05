from unittest import TestCase
from nearest_neighbor import *
import numpy as np


class TestNearestNeighbor(TestCase):
    def test_nearest_neighbor(self):
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
