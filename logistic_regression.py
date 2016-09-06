import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


class LogisticRegression:
    def __init__(self, X, y):
        def cost(theta):
            p = expit(theta @ X.T)
            return -np.sum(np.log(p ** y * (1.0 - p) ** (1 - y)))

        self.theta = minimize(cost, np.zeros(X.shape[1])).x

    def predict(self, X):
        return self.theta @ X.T > 0.0
