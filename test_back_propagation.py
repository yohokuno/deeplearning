from unittest import TestCase
from back_propagation import *


class TestBackPropagation(TestCase):
    def test_unit(self):
        unit1 = Unit()
        unit2 = Unit()
        unit3 = Unit(unit1, unit2)
        self.assertSequenceEqual(unit3.get_parents(), [unit1, unit2])
        self.assertSequenceEqual(unit3.get_children(), [])
        self.assertSequenceEqual(unit1.get_parents(), [])
        self.assertSequenceEqual(unit1.get_children(), [unit3])
        self.assertSequenceEqual(unit2.get_parents(), [])
        self.assertSequenceEqual(unit2.get_children(), [unit3])

    def test_constant(self):
        self.assertEqual(Constant(2).evaluate(), 2)

    def test_sum(self):
        sum_unit = Sum(Constant(2), Constant(3))
        self.assertEqual(sum_unit.evaluate(), 5)

        gradient = sum_unit.get_gradient()
        self.assertEqual(len(gradient), 2)
        self.assertEqual(gradient[0].evaluate(), 1)
        self.assertEqual(gradient[1].evaluate(), 1)

    def test_product(self):
        product_unit = Product(Constant(2), Constant(3))
        self.assertEqual(product_unit.evaluate(), 6)

        gradient = product_unit.get_gradient()
        self.assertEqual(len(gradient), 2)
        self.assertEqual(gradient[0].evaluate(), 3)
        self.assertEqual(gradient[1].evaluate(), 2)

        self.assertEqual(Product(Constant(4)).get_gradient()[0].evaluate(), 1)

    def test_back_propagation(self):
        pass

    def test_build_grad(self):
        variable = Constant(2)
        self.assertEqual(build_grad(variable).evaluate(), 0)

        x = Constant(4)
        Sum(x, x)
        self.assertEqual(build_grad(x).evaluate(), 2)

        x = Constant(4)
        Product(x, x)
        self.assertEqual(build_grad(x).evaluate(), 8)

        x = Constant(4)
        Sum(x, Product(x, x))
        self.assertEqual(build_grad(x).evaluate(), 9)

        x = Constant(4)
        Product(Constant(3), Product(Constant(2), x))
        self.assertEqual(build_grad(x).evaluate(), 6)

        x = Constant(4)
        Product(x, Product(x, x))
        self.assertEqual(build_grad(x).evaluate(), 3 * 16)
