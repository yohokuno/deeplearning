import unittest
from back_propagation import *


class TestBackPropagation(unittest.TestCase):
    def test_forward(self):
        # Identify function
        inputs = [1]
        functions = [[lambda x: x, [0]]]
        output = forward(inputs, functions)
        self.assertEqual(output, 1)

        # Multiplication
        inputs = [2, 3]
        functions = [[lambda x, y: x * y, [0, 1]]]
        output = forward(inputs, functions)
        self.assertEqual(output, 6)

        # 1 + 2 * 3
        inputs = [1, 2, 3]
        functions = [[lambda x, y: x * y, [1, 2]], [lambda x, y: x + y, [0, 3]]]
        output = forward(inputs, functions)
        self.assertEqual(output, 7)
