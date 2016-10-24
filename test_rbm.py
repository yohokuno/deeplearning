from unittest import TestCase
from rbm import *


class TestRBM(TestCase):
    def test_binary(self):
        actual = RBM.binary(2)
        expected = np.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [1, 1]])
        np.testing.assert_almost_equal(actual, expected)

    def test_rbm(self):
        W = np.array([[1, 0], [0, 1]])
        b = np.array([0, 0])
        c = np.array([0, 0])
        rbm = RBM(2, 2, W, b, c)

        v = np.array([1, 0])
        h = np.array([1, 0])
        actual = rbm.energy(v, h)
        expected = -1
        self.assertAlmostEqual(actual, expected)

        v = np.array([0, 1])
        h = np.array([1, 0])
        actual = rbm.energy(v, h)
        expected = 0
        self.assertAlmostEqual(actual, expected)

        actual = rbm.partition()
        expected = 9.0
        self.assertAlmostEqual(actual, expected)

        v = np.array([1, 0])
        h = np.array([1, 0])
        actual = rbm.joint(v, h)
        expected = 0.30203131427322721
        self.assertAlmostEqual(actual, expected)

        v = np.array([1, 0])
        actual = rbm.hidden(v)
        expected = [0.7310585786300049, 0.5]
        np.testing.assert_almost_equal(actual, expected)
