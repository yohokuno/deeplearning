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
        np.testing.assert_almost_equal(Variable(np.array([1, 2])).evaluate(), np.array([1, 2]))

    def test_add(self):
        add = Add(Variable(2), Variable(3))
        self.assertEqual(add.evaluate(), 5)
        self.assertEqual(add.get_gradient(0).evaluate(), 1)
        self.assertEqual(add.get_gradient(1).evaluate(), 1)

        add = Add(Variable(np.array([1, 2])), Variable(np.array([3, 4])))
        np.testing.assert_almost_equal(add.evaluate(), np.array([4, 6]))
        np.testing.assert_almost_equal(add.get_gradient(0).evaluate(), np.ones(2))
        np.testing.assert_almost_equal(add.get_gradient(1).evaluate(), np.ones(2))

    def test_subtract(self):
        subtract = Subtract(Variable(2), Variable(3))
        self.assertEqual(subtract.evaluate(), -1)
        self.assertEqual(subtract.get_gradient(0).evaluate(), 1)
        self.assertEqual(subtract.get_gradient(1).evaluate(), -1)

        A = Variable(np.array([[5, 6], [7, 8]]))
        B = Variable(np.array([[1, 2], [3, 4]]))
        subtract = Subtract(A, B)
        np.testing.assert_almost_equal(subtract.evaluate(), np.array([[4, 4], [4, 4]]))
        np.testing.assert_almost_equal(subtract.get_gradient(0).evaluate(), np.ones(A.evaluate().shape))
        np.testing.assert_almost_equal(subtract.get_gradient(1).evaluate(), -np.ones(B.evaluate().shape))

    def test_multiply(self):
        multiply = Multiply(Variable(2), Variable(3))
        self.assertEqual(multiply.evaluate(), 6)
        self.assertEqual(multiply.get_gradient(0).evaluate(), 3)
        self.assertEqual(multiply.get_gradient(1).evaluate(), 2)

        multiply = Multiply(Variable(2), Variable(3), Variable(4))
        self.assertEqual(multiply.evaluate(), 24)
        self.assertEqual(multiply.get_gradient(0).evaluate(), 12)
        self.assertEqual(multiply.get_gradient(1).evaluate(), 8)
        self.assertEqual(multiply.get_gradient(2).evaluate(), 6)

        A = Variable(np.array([[1, 2], [3, 4]]))
        B = Variable(np.array([[5, 6], [7, 8]]))
        multiply = Multiply(A, B)
        np.testing.assert_almost_equal(multiply.evaluate(), np.array([[5, 12], [21, 32]]))
        np.testing.assert_almost_equal(multiply.get_gradient(0).evaluate(), B.evaluate())
        np.testing.assert_almost_equal(multiply.get_gradient(1).evaluate(), A.evaluate())

        A = Variable(np.array([1, 2]))
        B = Variable(np.array([3, 4]))
        C = Variable(np.array([5, 6]))
        multiply = Multiply(A, B, C)
        np.testing.assert_almost_equal(multiply.evaluate(), np.array([15, 48]))
        np.testing.assert_almost_equal(multiply.get_gradient(0).evaluate(), np.array([15, 24]))
        np.testing.assert_almost_equal(multiply.get_gradient(1).evaluate(), np.array([5, 12]))
        np.testing.assert_almost_equal(multiply.get_gradient(2).evaluate(), np.array([3, 8]))

    def test_power(self):
        power = Power(Variable(2), 3)
        self.assertEqual(power.evaluate(), 8)
        self.assertEqual(power.get_gradient(0).evaluate(), 12)

        power = Power(Variable(3), 2)
        self.assertEqual(power.evaluate(), 9)
        self.assertEqual(power.get_gradient(0).evaluate(), 6)

        power = Power(Variable(np.array([1, 2])), 3)
        np.testing.assert_almost_equal(power.evaluate(), np.array([1, 8]))
        np.testing.assert_almost_equal(power.get_gradient(0).evaluate(), np.array([3, 12]))

        power = Power(Variable(np.array([2, 3])), 2)
        np.testing.assert_almost_equal(power.evaluate(), np.array([4, 9]))
        np.testing.assert_almost_equal(power.get_gradient(0).evaluate(), np.array([4, 6]))

    def test_sum(self):
        sum = Sum(Variable(np.array([1, 2])))
        self.assertEqual(sum.evaluate(), 3)
        np.testing.assert_almost_equal(sum.get_gradient(0).evaluate(), np.ones(2))

    def test_matrix_mutiply(self):
        matrix_multiply = MatrixMultiply(Variable(np.array([1, 2])), Variable(np.array([3, 4])))
        self.assertEqual(matrix_multiply.evaluate(), 11)
        np.testing.assert_almost_equal(matrix_multiply.get_gradient(0).evaluate(), np.array([3, 4]))
        np.testing.assert_almost_equal(matrix_multiply.get_gradient(1).evaluate(), np.array([1, 2]))

        matrix_multiply = MatrixMultiply(Variable(np.array([[1, 2], [3, 4]])), Variable(np.array([5, 6])))
        np.testing.assert_almost_equal(matrix_multiply.evaluate(), np.array([17, 39]))
        np.testing.assert_almost_equal(matrix_multiply.get_gradient(0).evaluate(), np.array([5, 6]))
        np.testing.assert_almost_equal(matrix_multiply.get_gradient(1).evaluate(), np.array([[1, 3], [2, 4]]))

    def test_relu(self):
        relu = Relu(Variable(3))
        self.assertEqual(relu.evaluate(), 3)
        self.assertEqual(relu.get_gradient(0).evaluate(), 1)

        relu = Relu(Variable(-3))
        self.assertEqual(relu.evaluate(), 0)
        self.assertEqual(relu.get_gradient(0).evaluate(), 0)

        relu = Relu(Variable(np.array([1, -2])))
        np.testing.assert_almost_equal(relu.evaluate(), np.array([1, 0]))
        np.testing.assert_almost_equal(relu.get_gradient(0).evaluate(), np.array([1, 0]))

        relu = Relu(Variable(np.array([[1, -2], [-3, 4]])))
        np.testing.assert_almost_equal(relu.evaluate(), np.array([[1, 0], [0, 4]]))

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

        x = Variable(np.array([1, 2]))
        y = Variable(3) * x
        np.testing.assert_almost_equal(differentiate(y, x).evaluate(), np.array([3, 3]))

        x = Variable(np.array([1, 2]))
        y = x + x * x
        np.testing.assert_almost_equal(differentiate(y, x).evaluate(), np.array([3, 5]))

    def test_linear_regression(self):
        x = [Variable(0), Variable(1)]
        y = [Variable(0), Variable(1)]
        w = Variable(0)
        f = [w * x[0], w * x[1]]
        J = (y[0] - f[0]) ** 2 + (y[1] - f[1]) ** 2
        dw = differentiate(J, w)

        for i in range(10):
            w_new = w.evaluate() - 0.5 * dw.evaluate()
            w.set_value(w_new)

        self.assertAlmostEqual(w.evaluate(), 1)
        self.assertAlmostEqual(J.evaluate(), 0)

        x = Variable(np.array([0, 1]))
        y = Variable(np.array([0, 1]))
        w = Variable(0)
        f = w * x
        J = Sum((y - f) ** 2)
        dw = differentiate(J, w)

        for i in range(10):
            w_new = w.evaluate() - 0.5 * dw.evaluate()
            w.set_value(w_new)

        self.assertAlmostEqual(w.evaluate(), 1)
        self.assertAlmostEqual(J.evaluate(), 0)

    def test_xor(self):
        X = Variable(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        y = Variable(np.array([0, 1, 1, 0]))
        W = Variable(np.array([[1, 1], [1, 1]]))
        c = Variable(np.array([0, -1]))
        w = Variable(np.array([1, -2]))

        np.testing.assert_almost_equal((X @ W).evaluate(), np.array([[0, 0], [1, 1], [1, 1], [2, 2]]))
        np.testing.assert_almost_equal((X @ W + c).evaluate(), np.array([[0, -1], [1, 0], [1, 0], [2, 1]]))
        np.testing.assert_almost_equal(Relu(X @ W + c).evaluate(), np.array([[0, 0], [1, 0], [1, 0], [2, 1]]))
        np.testing.assert_almost_equal((Relu(X @ W + c) @ w).evaluate(), np.array([0, 1, 1, 0]))

    def test_multi_layer_perceptron(self):
        X = Variable(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        y = Variable(np.array([0, 1, 1, 0]))
        #W1 = Variable(np.random.rand(2, 2))
        W1 = Variable(np.array([[1, 1], [1, 1]]))
        #c = Variable(np.random.rand(2))
        c = Variable(np.array([0, -1]))
        w2 = Variable(np.random.rand(2))
        p = Relu(X @ W1 + c) @ w2
        J = Sum((y - p) ** 2)
        dw2 = differentiate(J, w2)
        dc = differentiate(J, c)
        dW1 = differentiate(J, W1)

        learning_rate = 0.5
        for i in range(10):
            w2.set_value(w2.evaluate() - learning_rate * dw2.evaluate())
 #           c.set_value(c.evaluate() - learning_rate * dc.evaluate())
 #           W1.set_value(W1.evaluate() - learning_rate * dW1.evaluate())

        self.assertAlmostEqual(J.evaluate(), 0)
        np.testing.assert_almost_equal(W1.evaluate(), np.array([[1, 1], [1, 1]]))
        np.testing.assert_almost_equal(c.evaluate(), np.array([0, -1]))
        np.testing.assert_almost_equal(w2.evaluate(), np.array([1, -2]))
