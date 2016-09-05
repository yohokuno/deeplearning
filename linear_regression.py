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
        return X ** np.arange(0 if self.bias else 1, self.degree + 1)


class HyperLinearRegression:
    def __init__(self, X, y, max_degree=100):
        for d in range(max_degree):
            self.model = LinearRegression(X, y, False, d)
            if self.model.error(X, y) == 0.0:
                break

    def predict(self, X):
        return self.model.predict(X)

    def error(self, X, y):
        return self.model.error(X, y)
