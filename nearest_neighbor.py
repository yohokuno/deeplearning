import numpy as np


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
