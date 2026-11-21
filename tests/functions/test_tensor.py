"""Reshaping, indexing, reductions, and matrix multiplication."""

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


@pytest.mark.parametrize(
    'indices, expected_grad',
    [
        pytest.param(1, [[0, 0, 0], [1, 1, 1]], id='row'),
        pytest.param((1, 2), [[0, 0, 0], [0, 0, 1]], id='scalar'),
        pytest.param((slice(None), slice(None, None, 2)),
                     [[1, 0, 1], [1, 0, 1]], id='strided-columns'),
        pytest.param((Ellipsis, -1), [[0, 0, 1], [0, 0, 1]], id='last-column'),
        pytest.param([1, 0, 1], [[1, 1, 1], [2, 2, 2]], id='repeated-rows'),
        pytest.param((np.array([0, 1, 1]), np.array([2, 0, 0])),
                     [[0, 0, 1], [2, 0, 0]], id='repeated-pairs'),
        pytest.param(np.array([[True, False, True], [False, True, False]]),
                     [[1, 0, 1], [0, 1, 0]], id='boolean-mask'),
        pytest.param(slice(0, 0), [[0, 0, 0], [0, 0, 0]], id='empty'),
    ],
)
def test_get_item_forward_and_backward(indices, expected_grad):
    data = np.arange(6.0).reshape(2, 3)
    x = Variable(data)

    y = F.get_item(x, indices)
    y.sum().backward()

    np.testing.assert_array_equal(y.data, data[indices])
    np.testing.assert_array_equal(x.grad.data, expected_grad)
    assert x.grad.shape == x.shape
    assert x.grad.dtype == x.dtype


def test_get_item_accumulates_nonuniform_gradients_at_repeated_indices():
    x = Variable(np.array([2.0, 3.0, 5.0], dtype=np.float32))
    y = F.get_item(x, [2, 0, 2])
    y.grad = Variable(np.array([1.0, 2.0, 4.0], dtype=np.float32))

    y.backward()

    np.testing.assert_array_equal(x.grad.data, [2.0, 0.0, 5.0])
    assert x.grad.dtype == np.float32


def test_get_item_preserves_second_order_gradients_with_repeated_indices():
    x = Variable(np.array([2.0, 3.0, 5.0]))
    loss = (F.get_item(x, [0, 0, 2]) ** 3).sum()
    loss.backward(create_graph=True)
    first_grad = x.grad
    np.testing.assert_allclose(first_grad.data, [24.0, 0.0, 75.0])
    x.cleargrad()

    first_grad.sum().backward()

    np.testing.assert_allclose(x.grad.data, [24.0, 0.0, 30.0])


def test_broadcast_to_sums_gradients_back_to_original_shape():
    x = Variable(np.array([[2., 3., 4.]]))
    y = F.broadcast_to(x, (2, 3))
    weights = np.arange(6.).reshape(2, 3)
    (y * weights).sum().backward()
    np.testing.assert_array_equal(y.data, np.tile(x.data, (2, 1)))
    np.testing.assert_array_equal(x.grad.data, weights.sum(axis=0, keepdims=True))
    assert F.broadcast_to(x, x.shape) is x
    assert F.sum_to(x, x.shape) is x


def test_mse_normalizes_by_batch_and_differentiates_both_inputs():
    x = Variable(np.array([[1., 3.], [2., 4.]]))
    t = Variable(np.array([[0., 1.], [1., 2.]]))
    loss = F.mean_squared_error(x, t)
    (loss * 3).backward()
    np.testing.assert_allclose(loss.data, 5.)
    np.testing.assert_array_equal(x.grad.data, 3 * (x.data - t.data))
    np.testing.assert_array_equal(t.grad.data, -x.grad.data)


@pytest.mark.parametrize('bias', [False, True])
def test_linear_weighted_backward_matches_matrix_derivatives(bias):
    x = Variable(np.arange(6.).reshape(2, 3))
    w = Variable(np.arange(6.).reshape(3, 2))
    b = Variable(np.array([1., -2.])) if bias else None
    weights = np.array([[1., 2.], [3., 4.]])
    y = F.linear(x, w, b)
    (y * weights).sum().backward()
    np.testing.assert_allclose(y.data, x.data @ w.data + (b.data if bias else 0))
    np.testing.assert_allclose(x.grad.data, weights @ w.data.T)
    np.testing.assert_allclose(w.grad.data, x.data.T @ weights)
    if bias:
        np.testing.assert_allclose(b.grad.data, weights.sum(axis=0))


def test_explicit_transpose_axes_and_backward():
    x = Variable(np.arange(24.).reshape(2, 3, 4))
    weights = np.arange(24.).reshape(4, 2, 3)
    y = x.transpose(2, 0, 1)
    (y * weights).sum().backward()
    np.testing.assert_array_equal(y.data, np.transpose(x.data, (2, 0, 1)))
    np.testing.assert_array_equal(x.grad.data, weights.transpose(1, 2, 0))
