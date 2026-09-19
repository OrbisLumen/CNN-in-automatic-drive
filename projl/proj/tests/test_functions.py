import numpy as np

from mydezero import Variable, add, mul, exp, square


def test_square_forward():
    x = Variable(np.array(10.0))

    y = square(x)

    np.testing.assert_allclose(y.data, np.array(100.0))


def test_add_forward():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))

    y = add(x0, x1)

    np.testing.assert_allclose(y.data, np.array(5.0))


def test_composed_function_forward():
    x = Variable(np.array(0.5))

    y = square(exp(square(x)))

    np.testing.assert_allclose(y.data, 1.648721270700128)

def test_add_and_mul_forward():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = add(mul(a, b), c)
    np.testing.assert_allclose(y.data,7.0)
