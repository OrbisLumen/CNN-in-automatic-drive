import unittest
import numpy as np
from MyFlows import Variable, square, exp, add, mul
from MyFlows.numerical import numerical_diff, gradcheck


class TestLesson3(unittest.TestCase):
    def test_forward(self):
        x = Variable(0.5)
        y = square(exp(square(x)))
        self.assertAlmostEqual(float(y.data), np.exp(0.5))
        self.assertIsNotNone(y.creator)
    def test_manual_seed(self):
        with self.assertRaises(ValueError):
            square(Variable(0.5)).backward()
    def test_chain(self):
        for value in (0.0, 0.5, 2.0):
            x = Variable(value)
            y = square(exp(square(x)))
            y.grad = np.ones_like(y.data)
            y.backward()
            np.testing.assert_allclose(x.grad, 4 * value * np.exp(2 * value ** 2))
    def test_numerical(self):
        self.assertAlmostEqual(float(numerical_diff(square, Variable(2.))), 4., places=6)
    def test_gradcheck(self):
        for value in (0., 0.5, 2.):
            self.assertTrue(gradcheck(lambda x: square(exp(square(x))), Variable(value))[0])
    def test_tree(self):
        a, b, c = Variable(5.), Variable(3.), Variable(2.)
        y = mul(3., add(a, mul(b, c)))
        y.grad = np.ones_like(y.data)
        y.backward()
        self.assertEqual([float(v.grad) for v in (a,b,c)], [3.,6.,9.])


if __name__ == "__main__": unittest.main()
