import numpy as np


class LinearRegression:
    def __init__(self, X, y, bias=False, degree=1):
        self.bias = bias
        self.degree = degree
        X = self.preprocess(X)
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        X = self.preprocess(X)
        return self.w @ X.T

    def error(self, X, y):
        return np.average((self.predict(X) - y) ** 2)

    def preprocess(self, X):
        X = X ** np.arange(1, self.degree + 1)

        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X

