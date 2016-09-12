from unittest import TestCase
from back_propagation import *


class TestBackPropagation(TestCase):
    def test_back_propagation(self):
        pass

    def test_build_grad(self):
        input = Constant()
        output = Sum(input)
        grad_table = dict()
#        grad_table[output] = Constant()
        gradient = build_grad(input, output, grad_table)
        self.assertEqual(grad_table[input], Sum(input))
        self.assertEqual(gradient, Sum(input))
