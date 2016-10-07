import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def one_hot(y, category=None):
    if category is None:
        category = np.max(y) + 1
    result = np.zeros((y.size, category))
    result[np.arange(y.size), y] = 1
    return result


def rnn_predict(X, U, V, W, h_init):
    h = h_init

    for x in X:
        h = np.tanh(W @ h + U @ x)
        y = softmax(V @ h)
        yield y


def rnn_loss(X, U, V, W, h_init):
    h = h_init

    for i in range(len(X)-1):
        x = X[i]
        y_ = X[i+1]
        h = np.tanh(W @ h + U @ x)
        y = softmax(V @ h)
        yield np.sum(-y_ * np.log(y))
