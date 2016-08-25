from unittest import TestCase
from linear_algebra import *


class TestLinearAlgebra(TestCase):
    def test_transpose(self):
        A = [[1, 2], [3, 4]]
        actual = transpose(A)
        expected = [[1, 3], [2, 4]]
        self.assertEqual(actual, expected)

    def test_plus(self):
        A = [[1, 2], [3, 4]]
        B = [[1, 1], [1, 1]]
        actual = add(A, B)
        expected = [[2, 3], [4, 5]]
        self.assertEqual(actual, expected)

    def test_multiply(self):
        A = [[1, 2], [3, 4]]
        B = [[1, 1], [1, 1]]
        actual = multiply(A, B)
        expected = [[3, 3], [7, 7]]
        self.assertEqual(actual, expected)

    def test_identity(self):
        A = [[1, 2], [3, 4]]
        I = identity(2)
        self.assertEqual(multiply(A, I), A)
        self.assertEqual(multiply(I, A), A)
