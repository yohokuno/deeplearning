from unittest import TestCase
from logistic_regression import *


class TestLogisticRegression(TestCase):
    def test_logistic_regression(self):
        X = np.array([[1.0, 1.0],
                     [-1.0, -1.0]])
        y = np.array([1, 0])

        model = LogisticRegression(X, y)
        prediction = model.predict(X)
        np.testing.assert_almost_equal(prediction, y)
