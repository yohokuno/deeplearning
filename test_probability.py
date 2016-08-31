from unittest import TestCase
from probability import *


class TestProbability(TestCase):
    def test_is_probability(self):
        self.assertTrue(is_probability([1.0, 0.0]))
        self.assertTrue(is_probability([0.5, 0.5]))
        self.assertFalse(is_probability([0.0, 0.0]))

    def test_marginalize(self):
        self.assertEqual(marginalize([[0.1, 0.4], [0.3, 0.2]]), [0.5, 0.5])

    def test_condition(self):
        self.assertEqual(condition([[0.1, 0.4], [0.3, 0.2]]), [[0.2, 0.8], [0.6, 0.4]])
