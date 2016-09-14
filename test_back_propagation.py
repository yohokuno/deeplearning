from unittest import TestCase
from back_propagation import *


class TestBackPropagation(TestCase):
    def test_unit(self):
        unit1 = Unit()
        unit2 = Unit()
        unit3 = Unit(unit1, unit2)
        self.assertSequenceEqual(unit1.get_parents(), [])
        self.assertSequenceEqual(unit1.get_children(), [unit3])
        self.assertSequenceEqual(unit2.get_parents(), [])
        self.assertSequenceEqual(unit2.get_children(), [unit3])
        self.assertSequenceEqual(unit3.get_parents(), [unit1, unit2])
        self.assertSequenceEqual(unit3.get_children(), [])

    def test_variable(self):
        self.assertEqual(Variable(2).evaluate(), 2)

    def test_sum(self):
        variable1 = Variable(2)
        variable2 = Variable(3)
        sum_unit = Sum(variable1, variable2)
        self.assertEqual(sum_unit.evaluate(), 5)
        self.assertEqual(sum_unit.get_gradient(variable1).evaluate(), 1)
        self.assertEqual(sum_unit.get_gradient(variable2).evaluate(), 1)

    def test_product(self):
        variable = Variable(2)
        product = Product(variable)
        self.assertEqual(product.evaluate(), 2)
        self.assertEqual(product.get_gradient(variable).evaluate(), 1)

        variable1 = Variable(2)
        variable2 = Variable(3)
        product = Product(variable1, variable2)
        self.assertEqual(product.evaluate(), 6)
        self.assertEqual(product.get_gradient(variable1).evaluate(), 3)
        self.assertEqual(product.get_gradient(variable2).evaluate(), 2)

        variable1 = Variable(2)
        variable2 = Variable(3)
        variable3 = Variable(4)
        product = Product(variable1, variable2, variable3)
        self.assertEqual(product.evaluate(), 24)
        self.assertEqual(product.get_gradient(variable1).evaluate(), 12)
        self.assertEqual(product.get_gradient(variable2).evaluate(), 8)
        self.assertEqual(product.get_gradient(variable3).evaluate(), 6)

    def test_differentiate(self):
        x = Variable(3)
        y = Variable(2)
        self.assertEqual(differentiate(y, x).evaluate(), 0)

        x = Variable(3)
        y = Product(Variable(4), x)
        self.assertEqual(differentiate(y, x).evaluate(), 4)

        x = Variable(3)
        y = Product(x, x)
        self.assertEqual(differentiate(y, x).evaluate(), 6)

        x = Variable(3)
        y = Sum(x, Product(x, x))
        self.assertEqual(differentiate(y, x).evaluate(), 7)

        x = Variable(3)
        y = Product(x, Product(x, x))
        self.assertEqual(differentiate(y, x).evaluate(), 27)
