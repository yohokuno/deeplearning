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
        U = np.array([[0, 1], [1, 0]])
        V = np.array([[1, 0], [0, 1]])
        W = np.array([[0, 0], [0, 0]])
        h_init = np.array([0, 0])
        Y = list(rnn_loss(X, U, V, W, h_init))
        np.testing.assert_almost_equal(Y[0], Y[2])
        np.testing.assert_array_less(Y[1], Y[0])
