import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class LogisticRegression:
    def __init__(self, X, y):
        def cost(theta):
            p = expit(theta @ X.T)
            return -np.sum(np.log(p ** y * (1.0 - p) ** (1 - y)))

        self.theta = minimize(cost, np.zeros(X.shape[1])).x

    def predict(self, X):
        return self.theta @ X.T > 0.0


class SupportVectorMachine:
    def __init__(self, X, y, kernel=lambda x, y: np.exp(-np.sum((x - y)**2)*100)):
        def cost(alpha):
            predictions = np.sum(alpha * kernel(X[i], X[j]) for i in range(X.shape[0]) for j in range(X.shape[1]))
            return np.sum((y - predictions) ** 2)

        self.kernel = kernel
        self.X = X
        self.alpha = minimize(cost, np.zeros(X.shape[0])).x

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = np.sum(self.alpha[i] * self.kernel(self.X[i], x) for i in range(self.X.shape[0]))
            predictions.append(prediction)
        return np.sign(predictions)
