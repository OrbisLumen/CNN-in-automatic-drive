"""Reshaping, reductions, and matrix multiplication."""

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F


def test_reshape_backward_restores_input_shape():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)

    np.testing.assert_array_equal(x.grad.data, np.ones_like(x.data))


def test_transpose_backward_restores_input_shape():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward()

    np.testing.assert_array_equal(x.grad.data, np.ones_like(x.data))


def test_sum_along_axis_forward_and_backward():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()

    np.testing.assert_array_equal(y.data, [5, 7, 9])
    np.testing.assert_array_equal(x.grad.data, np.ones_like(x.data))


def test_variable_sum_keeps_reduced_dimensions():
    x = Variable(np.arange(120.0).reshape(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    assert y.shape == (1, 1, 1, 1)


@pytest.mark.parametrize(
    'shape, expected',
    [
        pytest.param((1, 3), [[5, 7, 9]], id='reduce-rows'),
        pytest.param((2, 1), [[6], [15]], id='reduce-columns'),
    ],
)
def test_sum_to_reduces_to_target_shape(shape, expected):
    x = np.array([[1, 2, 3], [4, 5, 6]])

    y = F.sum_to(x, shape)

    np.testing.assert_array_equal(y.data, expected)


def test_matmul_forward_and_backward():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    W = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
    y = F.matmul(x, W)
    y.backward()

    np.testing.assert_array_equal(y.data, [[22, 28], [49, 64]])
    np.testing.assert_array_equal(x.grad.data, [[3, 7, 11], [3, 7, 11]])
    np.testing.assert_array_equal(W.grad.data, [[5, 5], [7, 7], [9, 9]])
