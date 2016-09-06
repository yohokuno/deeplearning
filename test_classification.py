from unittest import TestCase
from classification import *


class TestClassification(TestCase):
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
