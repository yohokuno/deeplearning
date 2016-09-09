import unittest
from back_propagation import *


class TestBackPropagation(unittest.TestCase):
    def test_back_propagation(self):
        inputs = [2]
        functions = [identity(0)]
        gradients = back_propagation(inputs, functions)
        self.assertEqual(gradients, [1, 1])

        inputs = [2, 3]
        functions = [multiply(0, 1)]
        gradients = back_propagation(inputs, functions)
        self.assertEqual(gradients, [3, 2, 1])

        inputs = [2, 3, 4]
        functions = [multiply(0, 1), add(2, 3)]
        gradients = back_propagation(inputs, functions)
        self.assertEqual(gradients, [3, 2, 1, 1, 1])
