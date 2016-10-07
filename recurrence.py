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


def rnn_loss(X, Y_, U, V, W, h_init):
    h = h_init

    for i in range(len(X)):
        h = np.tanh(W @ h + U @ X[i])
        y = softmax(V @ h)
        yield np.sum(-Y_[i] * np.log(y))


def rnn_gradient(X, Y_, U, V, W, h_init):
    # Forward
    H = [h_init]
    Y = []

    for i in range(len(X)):
        h = np.tanh(W @ H[i] + U @ X[i])
        y = softmax(V @ h)
        H.append(h)
        Y.append(y)

    # Backward
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    dh = V.T @ (Y[-1] - Y_[-1])

    for i in range(len(X) - 2, -1, -1):
        do = Y[i] - Y_[i]
        dh = W.T @ dh @ np.diag(1 - H[i+2] ** 2) + V.T @ do
        dV += np.outer(do, H[i+1])
        dW += np.outer(np.diag(1 - H[i+1] ** 2) @ dh, H[i])
        dU += np.outer(np.diag(1 - H[i+1] ** 2) @ dh, X[i])

    return dV, dW, dU
