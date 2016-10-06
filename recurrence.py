import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def one_hot(y, category=None):
    if category is None:
        category = np.max(y) + 1
    result = np.zeros((y.size, category))
    result[np.arange(y.size), y] = 1
    return result


def recurrence(X, U, V, W, h_init):
    h = h_init

    for x in X:
        h = np.tanh(W @ h + U @ x)
        y = softmax(V @ h)
        yield y
