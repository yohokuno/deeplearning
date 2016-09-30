from unittest import TestCase
from convolution import *


class TestConvolution(TestCase):
    def test_convolution1d_function(self):
        function = convolution1d_function(lambda t: t, lambda t: t)
        actual = function(0, range(5))
        self.assertEqual(actual, -30)

        actual = function(1, range(5))
        self.assertEqual(actual, -20)

        actual = function(0, range(-2, 3))
        self.assertEqual(actual, -10)

    def test_convolution2d_function(self):
        image = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        kernel = [[0, 1], [1, 0]]
        function = convolution2d_function(lambda i, j: image[i][j], lambda m, n: kernel[m][n])
        actual = function(1, 1, range(2), range(2))
        self.assertEqual(actual, 4)

        actual = function(2, 2, range(2), range(2))
        self.assertEqual(actual, 12)

    def test_cross_correlation_function(self):
        image = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        kernel = [[0, 1], [1, 0]]
        function = cross_correlation_function(lambda i, j: image[i][j], lambda m, n: kernel[m][n])
        actual = function(0, 0, range(2), range(2))
        self.assertEqual(actual, 4)

        actual = function(1, 1, range(2), range(2))
        self.assertEqual(actual, 12)

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
