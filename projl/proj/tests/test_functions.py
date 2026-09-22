import numpy as np

from mydezero import Variable
from mydezero.core import add
import mydezero.functions as F


def test_square_forward():
    x = Variable(np.array(10.0))

    y = F.square(x)

    np.testing.assert_allclose(y.data, np.array(100.0))


def test_add_forward():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))

    y = add(x0, x1)

    np.testing.assert_allclose(y.data, np.array(5.0))


def test_composed_function_forward():
    x = Variable(np.array(0.5))

    y = F.square(F.exp(F.square(x)))

    np.testing.assert_allclose(y.data, 1.648721270700128)


def test_basic_sin():
    x = Variable(np.array(np.array(np.pi / 4)))
    y = F.sin(x)
    y.backward()

    np.testing.assert_allclose(y.data, 0.7071067811865476)
    np.testing.assert_allclose(x.grad.data, 0.7071067811865476)

def test_sin_higher_order_derivative():
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    grads = []
    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        grads.append(x.grad.data)

    np.testing.assert_allclose(grads[0], -0.8414709848078965)
    np.testing.assert_allclose(grads[1], -0.5403023058681398)
    np.testing.assert_allclose(grads[2], 0.8414709848078965)
