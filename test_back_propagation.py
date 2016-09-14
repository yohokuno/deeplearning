from unittest import TestCase
from back_propagation import *


class TestBackPropagation(TestCase):
    def test_unit(self):
        unit1 = Unit()
        unit2 = Unit()
        unit3 = Unit(unit1, unit2)
        self.assertSequenceEqual(unit1.get_parents(), [])
        self.assertSequenceEqual(unit1.get_children(), [(unit3, 0)])
        self.assertSequenceEqual(unit2.get_parents(), [])
        self.assertSequenceEqual(unit2.get_children(), [(unit3, 1)])
        self.assertSequenceEqual(unit3.get_parents(), [unit1, unit2])
        self.assertSequenceEqual(unit3.get_children(), [])

    def test_variable(self):
        self.assertEqual(Variable(2).evaluate(), 2)

    def test_sum(self):
        variable1 = Variable(2)
        variable2 = Variable(3)
        sum_unit = Sum(variable1, variable2)
        self.assertEqual(sum_unit.evaluate(), 5)
        self.assertEqual(sum_unit.get_gradient(0).evaluate(), 1)
        self.assertEqual(sum_unit.get_gradient(1).evaluate(), 1)

        sum_unit = variable1 + variable2 + Variable(4)
        self.assertEqual(sum_unit.evaluate(), 9)
        self.assertEqual(sum_unit.get_gradient(0).evaluate(), 1)
        self.assertEqual(sum_unit.get_gradient(1).evaluate(), 1)
        self.assertEqual(sum_unit.get_gradient(2).evaluate(), 1)

    def test_product(self):
        variable = Variable(2)
        product = Product(variable)
        self.assertEqual(product.evaluate(), 2)
        self.assertEqual(product.get_gradient(0).evaluate(), 1)

        variable1 = Variable(2)
        variable2 = Variable(3)
        product = Product(variable1, variable2)
        self.assertEqual(product.evaluate(), 6)
        self.assertEqual(product.get_gradient(0).evaluate(), 3)
        self.assertEqual(product.get_gradient(1).evaluate(), 2)

        variable1 = Variable(2)
        variable2 = Variable(3)
        variable3 = Variable(4)
        product = Product(variable1, variable2, variable3)
        self.assertEqual(product.evaluate(), 24)
        self.assertEqual(product.get_gradient(0).evaluate(), 12)
        self.assertEqual(product.get_gradient(1).evaluate(), 8)
        self.assertEqual(product.get_gradient(2).evaluate(), 6)

        variable1 = Variable(2)
        variable2 = Variable(3)
        product = Product(variable1, variable2)
        self.assertEqual(product.evaluate(), 6)
        self.assertEqual(product.get_gradient(0).evaluate(), 3)
        self.assertEqual(product.get_gradient(1).evaluate(), 2)

    def test_differentiate(self):
        x = Variable(3)
        y = Variable(2)
        self.assertEqual(differentiate(y, x).evaluate(), 0)

        x = Variable(3)
        y = Variable(4) * x
        self.assertEqual(differentiate(y, x).evaluate(), 4)

        x = Variable(3)
        y = x * x
        self.assertEqual(differentiate(y, x).evaluate(), 6)

        x = Variable(3)
        y = x + x * x
        self.assertEqual(differentiate(y, x).evaluate(), 7)

        x = Variable(3)
        y = x * x * x
        derivative = differentiate(y, x)
        self.assertEqual(derivative.evaluate(), 27)
        second_derivative = differentiate(derivative, x)
        self.assertEqual(second_derivative.evaluate(), 18)

    def test_linear_regression(self):
        x = [Variable(0), Variable(1)]
        y = [Variable(0), Variable(1)]
        w = Variable(0)
        f = [w * x[0], w * x[1]]
        J = (y[0] + Variable(-1) * f[0]) * (y[0] + Variable(-1) * f[0]) + (y[1] + Variable(-1) * f[1]) * (y[1] + Variable(-1) * f[1])
        dw = differentiate(J, w)

        for i in range(10):
            w_new = w.evaluate() - 0.5 * dw.evaluate()
            w.set_value(w_new)

        self.assertAlmostEqual(w.evaluate(), 1)
        self.assertAlmostEqual(J.evaluate(), 0)
