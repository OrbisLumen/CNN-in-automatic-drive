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


def test_cos_forward_and_backward():
    x = Variable(np.array([-1.2, 0., 0.7]))
    y = F.cos(x)
    y.sum().backward()
    np.testing.assert_allclose(y.data, np.cos(x.data))
    np.testing.assert_allclose(x.grad.data, -np.sin(x.data))


def test_sigmoid_simple_matches_optimized_value_and_gradient():
    x = Variable(np.array([-2., 0., 3.]))
    y = F.sigmoid_simple(x)
    y.sum().backward()
    expected = 1 / (1 + np.exp(-x.data))
    np.testing.assert_allclose(y.data, expected)
    np.testing.assert_allclose(x.grad.data, expected * (1 - expected))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_relu_forward_backward_zero_boundary_and_input_preservation(dtype):
    data = np.array([[-2., 0., 3.], [1., -1., 0.]], dtype=dtype)
    original = data.copy()
    x = Variable(data)
    y = F.relu(x)
    weights = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=dtype)
    (y * weights).sum().backward()
    np.testing.assert_array_equal(y.data, [[0, 0, 3], [1, 0, 0]])
    np.testing.assert_array_equal(x.grad.data, [[0, 0, 3], [4, 0, 0]])
    np.testing.assert_array_equal(data, original)
    assert y.shape == x.shape and y.dtype == dtype
    assert x.grad.dtype == dtype


def test_relu_second_derivative_of_squared_output():
    x = Variable(np.array([-2., 0., 3.]))
    (F.relu(x) ** 2).sum().backward(create_graph=True)
    first = x.grad
    x.cleargrad()
    first.sum().backward()
    np.testing.assert_array_equal(x.grad.data, [0, 0, 2])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dropout_mask_ratio_rescaling_and_dtype(dtype):
    np.random.seed(0)
    x = Variable(np.random.rand(100, 100).astype(dtype))

    y = F.dropout(x, 0.5)

    assert y.shape == x.shape and y.dtype == dtype
    kept = y.data != 0
    assert 0.45 < kept.mean() < 0.55
    np.testing.assert_allclose(y.data[kept], x.data[kept] / 0.5)


def test_dropout_with_zero_ratio_keeps_input():
    np.random.seed(0)
    x = Variable(np.random.rand(10, 10))

    y = F.dropout(x, 0.0)

    np.testing.assert_array_equal(y.data, x.data)
