import numpy as np


class LinearRegression:
    def __init__(self, X, y, bias):
        self.bias = bias
        if self.bias:
            X = self.add_bias(X)
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        if self.bias:
            X = self.add_bias(X)
        return self.w @ X.T

    def error(self, X, y):
        return np.average((self.predict(X) - y) ** 2)

    @staticmethod
    def add_bias(X):
        return np.hstack((X, np.ones((X.shape[0], 1))))

