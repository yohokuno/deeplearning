from unittest import TestCase
from recurrence import *


class TestRecurrence(TestCase):
    def test_softmax(self):
        actual = softmax(np.array([0, 1]))
        expected = np.array([0.2689414, 0.7310586])
        np.testing.assert_almost_equal(actual, expected)

    def test_one_hot(self):
        actual = one_hot(np.array([0, 1]))
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_almost_equal(actual, expected)

    def test_rnn_predict(self):
        # Bigram
        X = one_hot(np.array([0, 1, 0, 1]))
        U = np.array([[0, 1], [1, 0]])
        V = np.array([[1, 0], [0, 1]])
        W = np.array([[0, 0], [0, 0]])
        h_init = np.array([0, 0])
        Y = np.array(list(rnn_predict(X, U, V, W, h_init)))
        Y_ = np.array([1, 0, 1, 0])
        np.testing.assert_almost_equal(np.argmax(Y, axis=1), Y_)

        # Accumulate previous inputs
        X = one_hot(np.array([0, 0, 1, 1]))
        U = np.array([[1, 0], [0, 1]])
        V = np.array([[1, 0], [0, 1]])
        W = np.array([[2, 0], [0, 2]])
        h_init = np.array([0, 0])
        Y = np.array(list(rnn_predict(X, U, V, W, h_init)))
        Y_ = np.array([0, 0, 0, 1])
        np.testing.assert_almost_equal(np.argmax(Y, axis=1), Y_)

    def test_rnn_loss(self):
        X = one_hot(np.array([0, 0, 1, 1]))
        Y_ = one_hot(np.array([0, 1, 1, 1]))
        U = np.array([[0, 1], [1, 0]])
        V = np.array([[1, 0], [0, 1]])
        W = np.array([[0, 0], [0, 0]])
        h_init = np.array([0, 0])
        L = list(rnn_loss(X, Y_, U, V, W, h_init))
        np.testing.assert_almost_equal(L[0], L[2])
        np.testing.assert_array_less(L[1], L[0])

    def test_rnn_gradient(self):
        X = one_hot(np.array([0, 0, 1, 1]))
        Y_ = one_hot(np.array([0, 1, 1, 0]))
        U = np.array([[1., 0.], [0., 1.]])
        V = np.array([[1., 0.], [0., 1.]])
        W = np.array([[1., 0.], [0., 1.]])
        h_init = np.array([0, 0])

        for i in range(1000):
            dV, dW, dU = rnn_gradient(X, Y_, U, V, W, h_init)
            V -= 0.2 * dV
            W -= 0.2 * dW
            U -= 0.2 * dU

        Y = np.array(list(rnn_predict(X, U, V, W, h_init)))
        np.testing.assert_almost_equal(np.argmax(Y, axis=1), np.argmax(Y_, axis=1))
