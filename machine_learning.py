import numpy as np
from scipy.optimize import minimize
from scipy.special._ufuncs import expit


class LinearRegression:
    def __init__(self, X, y, bias=False, degree=1, l=0.0):
        self.bias = bias
        self.degree = degree
        X = self.preprocess(X)
        self.w = np.linalg.solve(X.T @ X + l * np.identity(X.shape[1]), X.T @ y)

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
            if np.isclose(self.model.error(X, y), 0.0):
                break

    def predict(self, X):
        return self.model.predict(X)

    def error(self, X, y):
        return self.model.error(X, y)


class NearestNeighbor:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        result = []

        for x in X:
            i = np.argmin((x - X) ** 2)
            result.append(self.y[i])

        return np.array(result)

    def error(self, X, y):
        return np.average((self.predict(X) - y) ** 2)


class LogisticRegression:
    def __init__(self, X, y):
        def cost(theta):
            p = expit(theta @ X.T)
            return -np.sum(np.log(p ** y * (1.0 - p) ** (1 - y)))

        self.theta = minimize(cost, np.zeros(X.shape[1])).x

    def predict(self, X):
        return self.theta @ X.T > 0.0


class SupportVectorMachine:
    def __init__(self, X, y, kernel=lambda x, y: np.exp(np.sum((x - y)**2))):
        def cost(alpha):
            predictions = np.sum(alpha * kernel(X[i], X[j]) for i in range(X.shape[0]) for j in range(X.shape[1]))
            return np.sum((y - predictions) ** 2)

        self.kernel = kernel
        self.X = X
        self.alpha = minimize(cost, np.zeros(X.shape[0])).x

    def predict(self, X):
        predictions = np.sum(self.alpha * self.kernel(X[i], X[j]) for i in range(X.shape[0]) for j in range(X.shape[1]))
        return predictions
