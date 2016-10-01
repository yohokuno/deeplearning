from unittest import TestCase
from convolution import *


class TestConvolution(TestCase):
    def test_convolution1d(self):
        I = [1, 2, 3]
        K = [1, -1]
        actual = convolution1d(I, K)
        expected = [-1, -1]
        self.assertSequenceEqual(actual, expected)

    def test_convolution2d(self):
        I = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        K = [[0, 1], [1, 0]]
        actual = convolution2d(I, K)
        expected = [[4, 6], [10, 12]]
        self.assertSequenceEqual(actual, expected)

    def test_max_pooling1d(self):
        I = [0, 0.1, 1, 0.2, 0.1, 0]
        actual = max_pooling1d(I, 2)
        expected = [0.1, 1, 1, 0.2, 0.1]
        self.assertSequenceEqual(actual, expected)

        I = [0, 0.1, 1, 0.2, 0.1, 0]
        actual = max_pooling1d(I, 3)
        expected = [1, 1, 1, 0.2]
        self.assertSequenceEqual(actual, expected)

        I = [0.1, 1, 0.2, 0.1, 0, 0.1, 0]
        actual = max_pooling1d(I, 3, 2)
        expected = [1, 0.2, 0.1]
        self.assertSequenceEqual(actual, expected)

    def test_max_pooling2d(self):
        I = [[0, 2, 1], [2, 1, 0], [0, 3, 2]]
        actual = max_pooling2d(I, 2)
        expected = [[2, 2], [3, 3]]
        self.assertSequenceEqual(actual, expected)

        actual = max_pooling2d(I, 2, 2)
        expected = [[2]]
        self.assertSequenceEqual(actual, expected)

    def test_max_pooling1d_fix(self):
        I = [0, 0.1, 1, 0.2, 0.1, 0]
        actual = max_pooling1d_fix(I, 2)
        expected = [1, 0.2]
        self.assertSequenceEqual(actual, expected)

    def test_max_pooling2d_fix(self):
        I = [[0, 2, 1, 0],
             [2, 1, 0, 2],
             [0, 3, 2, 1],
             [2, 1, 3, 2]]
        actual = max_pooling2d_fix(I, 2)
        expected = [[2, 2], [3, 3]]
        self.assertSequenceEqual(actual, expected)
