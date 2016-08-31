from unittest import TestCase
from probability import *


class TestProbability(TestCase):
    def test_is_probability(self):
        self.assertTrue(is_probability([1.0, 0.0]))
        self.assertTrue(is_probability([0.5, 0.5]))
        self.assertFalse(is_probability([0.0, 0.0]))
        self.assertFalse(is_probability([-1.0, 2.0]))

    def test_marginalize(self):
        self.assertEqual(marginalize([[0.1, 0.4], [0.3, 0.2]], axis=1), [0.5, 0.5])
        self.assertEqual(marginalize([[0.1, 0.4], [0.3, 0.2]], axis=0), [0.4, 0.6000000000000001])

    def test_condition(self):
        self.assertEqual(condition([[0.1, 0.4], [0.3, 0.2]]), [[0.2, 0.8], [0.6, 0.4]])

    def test_is_independent(self):
        self.assertTrue(is_independent([[0.1, 0.4], [0.1, 0.4]]))
        self.assertTrue(is_independent([[0.2, 0.2], [0.3, 0.3]]))
        self.assertFalse(is_independent([[0.1, 0.4], [0.4, 0.1]]))

    def test_expectation(self):
        self.assertEqual(expectation([0.1, 0.9], lambda x: x), 0.9)

    def test_uniform(self):
        self.assertEqual(uniform(2)(0), 0.5)

    def test_bernoulli(self):
        self.assertEqual(bernoulli(0.1)(0), 0.9)

    def test_multinoulli(self):
        self.assertEqual(multinoulli([0.1, 0.9])(1), 0.9)

    def test_gaussian(self):
        N = gaussian(0.0, 1.0)
        self.assertGreater(N(0.0), N(0.1))
        self.assertGreater(N(0.0), N(-0.1))
        self.assertEqual(N(0.1), N(-0.1))

    def test_mixture(self):
        G1 = gaussian(1.0, 1.0)
        G2 = gaussian(-1.0, 1.0)
        M = mixture([G1, G2], [0.6, 0.4])
        self.assertGreater(M(1.0), M(-1.0))
        self.assertGreater(M(1.0), M(0.0))
        self.assertLess(M(-10.0), M(0.0))
