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

        actual = multiply(2, A)
        expected = [[2, 4], [6, 8]]
        self.assertEqual(actual, expected)

        actual = multiply([1, 2], A)
        expected = [7, 10]
        self.assertEqual(actual, expected)

        actual = multiply(A, [1, 2])
        expected = [5, 11]
        self.assertEqual(actual, expected)

    def test_identity(self):
        A = [[1, 2], [3, 4]]
        I = identity(2)
        self.assertEqual(multiply(A, I), A)
        self.assertEqual(multiply(I, A), A)

    def test_inverse(self):
        A = [[1, 2], [3, 4]]
        A_inverse = inverse(A)
        self.assertEqual(multiply(A, A_inverse), identity(2))
        self.assertEqual(multiply(A_inverse, A), identity(2))

    def test_norm(self):
        x = [3, 4]
        actual = norm(x)
        expected = 5
        self.assertEqual(actual, expected)

    def test_diagonal(self):
        x = [1, 2]
        actual = diagonal(x)
        expected = [[1, 0], [0, 2]]
        self.assertEqual(actual, expected)

    def test_eigen_composition(self):
        V = [[1, 1], [1, -1]]
        l = [1, 2]
        A = eigen_composition(V, l)
        self.assertEqual(multiply(A, [1, 1]), [1, 1])
        self.assertEqual(multiply(A, [1, -1]), multiply(2, [1, -1]))
