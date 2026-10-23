"""Convolution, transposed convolution, and pooling correctness."""

import numpy as np

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
