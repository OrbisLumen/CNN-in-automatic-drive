"""Convolution, transposed convolution, and pooling correctness."""

import numpy as np
import pytest

from mydezero import Variable, no_grad
import mydezero.functions as F


def numerical_gradient(function, data, eps=1e-5):
    gradient = np.zeros_like(data)
    for index in np.ndindex(data.shape):
        above, below = data.copy(), data.copy()
        above[index] += eps
        below[index] -= eps
        with no_grad():
            gradient[index] = (function(above).data - function(below).data) / (2 * eps)
    return gradient


def test_conv2d_forward_shape():
    x = Variable(np.random.rand(2, 3, 8, 8).astype(np.float32))
    W = Variable(np.random.rand(4, 3, 3, 3).astype(np.float32))

    y = F.conv2d(x, W, b=None, stride=1, pad=1)

    assert y.shape == (2, 4, 8, 8)


def test_conv2d_backward_matches_numerical_gradient():
    np.random.seed(0)
    x_data = np.random.rand(1, 1, 4, 4).astype(np.float64)
    W_data = np.random.rand(1, 1, 3, 3).astype(np.float64)
    x = Variable(x_data.copy())
    W = Variable(W_data.copy())

    def objective(values):
        return F.conv2d(values, W, b=None, stride=1, pad=1).sum()

    objective(x).backward()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, x_data), atol=1e-6)


def test_pooling_forward_and_backward_route_to_argmax():
    data = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    x = Variable(data.copy())

    y = F.pooling(x, kernel_size=2, stride=2)
    y.sum().backward()

    np.testing.assert_array_equal(y.data, [[[[6.0, 8.0], [14.0, 16.0]]]])
    expected = np.zeros_like(data)
    expected[0, 0, 1, 1] = 1
    expected[0, 0, 1, 3] = 1
    expected[0, 0, 3, 1] = 1
    expected[0, 0, 3, 3] = 1
    np.testing.assert_array_equal(x.grad.data, expected)


def test_average_pooling_forward_and_backward():
    x = Variable(np.array([[[[1.0, 2.0], [3.0, 4.0]]]]))

    y = F.average_pooling(x, kernel_size=2, stride=2)
    y.sum().backward()

    np.testing.assert_array_equal(y.data, [[[[2.5]]]])
    np.testing.assert_allclose(x.grad.data, [[[[0.25, 0.25], [0.25, 0.25]]]])


def test_deconv2d_backward_matches_numerical_gradient():
    np.random.seed(0)
    x_data = np.random.rand(1, 2, 4, 4).astype(np.float64)
    W_data = np.random.rand(2, 3, 3, 3).astype(np.float64)
    x = Variable(x_data.copy())
    W = Variable(W_data.copy())

    def objective(values):
        return F.deconv2d(values, W, b=None, stride=2, pad=1).sum()

    objective(x).backward()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, x_data), atol=1e-6)


@pytest.mark.parametrize("operation", [F.conv2d, F.deconv2d])
def test_convolution_input_weight_and_bias_gradients(operation):
    rng = np.random.default_rng(0)
    x = Variable(rng.normal(size=(1, 2, 3, 4)))
    weight_shape = (3, 2, 2, 3) if operation is F.conv2d else (2, 3, 2, 3)
    weight = Variable(rng.normal(size=weight_shape))
    bias = Variable(rng.normal(size=3))
    output_weights = rng.normal(size=operation(x, weight, bias, (2, 1), (1, 1)).shape)

    def objective(values, parameter):
        args = [x, weight, bias]
        args[parameter] = values
        return (operation(*args, stride=(2, 1), pad=(1, 1)) * output_weights).sum()

    objective(x, 0).backward()

    for index, variable in enumerate((x, weight, bias)):
        expected = numerical_gradient(lambda values: objective(values, index), variable.data)
        np.testing.assert_allclose(variable.grad.data, expected, atol=1e-8)


def test_average_pooling_rectangular_kernel_backward():
    data = np.arange(1, 13, dtype=np.float64).reshape(1, 1, 3, 4)
    x = Variable(data.copy())
    weights = np.array([[[[1.0, -2.0], [3.0, 4.0]]]])

    def objective(values):
        return (F.average_pooling(values, (2, 3)) * weights).sum()

    np.testing.assert_array_equal(
        F.average_pooling(x, (2, 3)).data, [[[[4.0, 5.0], [8.0, 9.0]]]]
    )
    objective(x).backward()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, data), atol=1e-8)


def test_conv2d_weight_gradient_supports_second_derivatives():
    x = Variable(np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3))
    weight = Variable(np.ones((1, 1, 2, 2)))

    F.conv2d(x, weight).sum().backward(create_graph=True)

    weight_gradient = weight.grad
    np.testing.assert_array_equal(weight_gradient.data, [[[[12, 16], [24, 28]]]])
    x.cleargrad()
    weight_gradient.sum().backward()
    np.testing.assert_array_equal(x.grad.data, [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]])
