import numpy as np
from scipy.special import expit


class RBM:
    def __init__(self, v_size, h_size, W, b, c):
        self.v_size = v_size
        self.h_size = h_size
        self.W = W
        self.b = b
        self.c = c

    def energy(self, v, h):
        return -v.T @ self.W @ h - self.b @ v - self.c @ h

    def partition(self):
        v = self.__class__.binary(self.v_size).T
        h = self.__class__.binary(self.h_size).T
        return np.sum(np.exp(-self.energy(v, h)))

    def joint(self, v, h):
        return np.exp(-self.energy(v, h)) / self.partition()

    def hidden(self, v):
        return expit(self.c + self.W.T @ v)

    def visible(self, h):
        return expit(self.c + self.W.T @ h)

    @staticmethod
    def binary(size):
        x = np.arange(2 ** size, dtype=np.uint8)
        y = np.expand_dims(x, 1)
        z = np.unpackbits(y, 1)
        return z[:, 8 - size:]
