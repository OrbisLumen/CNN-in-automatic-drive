"""Elementary functions, activations, and their derivatives."""

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F


def test_square_forward():
    x = Variable(np.array(10.0))

    y = F.square(x)

    np.testing.assert_allclose(y.data, np.array(100.0))


def test_composed_function_forward():
    x = Variable(np.array(0.5))

    y = F.square(F.exp(F.square(x)))

    np.testing.assert_allclose(y.data, 1.648721270700128)


def test_exp_higher_order_derivative():
    x = Variable(np.array(1.0))
    y = F.exp(x)
    y.backward(create_graph=True)

    for _ in range(2):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        np.testing.assert_allclose(x.grad.data, np.e)


@pytest.mark.parametrize('value', [-1000.0, -1.0, 0.0, 1.0, 1000.0])
def test_sigmoid_forward_and_backward_without_overflow(value):
    x = Variable(np.array(value))
    with np.errstate(over='raise', invalid='raise'):
        y = F.sigmoid(x)
        y.backward()

    expected = np.exp(-np.logaddexp(0.0, -value))
    np.testing.assert_allclose(y.data, expected)
    np.testing.assert_allclose(x.grad.data, expected * (1 - expected))


def test_sin_forward_and_backward():
    x = Variable(np.array(np.pi / 4))
    y = F.sin(x)
    y.backward()

    np.testing.assert_allclose(y.data, 0.7071067811865476)
    np.testing.assert_allclose(x.grad.data, 0.7071067811865476)


def test_sin_higher_order_derivative():
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    grads = []
    for _ in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        grads.append(x.grad.data)

    np.testing.assert_allclose(
        grads, [-0.8414709848078965, -0.5403023058681398, 0.8414709848078965]
    )


def test_tanh_forward_and_backward():
    x = Variable(np.array(2.0))
    y = F.tanh(x)
    y.backward()

    np.testing.assert_allclose(x.grad.data, 0.07065082)
    np.testing.assert_allclose(y.data, 0.96402758)
