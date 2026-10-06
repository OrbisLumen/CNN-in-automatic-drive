"""Extrema reduction gradients and clipping boundaries."""

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F


@pytest.mark.parametrize('operation, reference', [(F.max, np.max), (F.min, np.min)])
@pytest.mark.parametrize('axis', [None, 0, 1, -1, (0, 2), (-3, -1), ()])
@pytest.mark.parametrize('keepdims', [False, True])
def test_extrema_forward_and_backward_restore_input_shape(operation, reference, axis, keepdims):
    data = np.arange(24.0).reshape(2, 3, 4)
    x = Variable(data.copy())
    y = operation(x, axis=axis, keepdims=keepdims)
    np.testing.assert_array_equal(y.data, reference(data, axis=axis, keepdims=keepdims))
    weights = np.asarray(np.arange(y.size, dtype=float).reshape(y.shape) + 1)
    y.grad = Variable(weights)

    y.backward()

    reduced = reference(data, axis=axis, keepdims=True)
    expected = (data == reduced) * weights.reshape(reduced.shape)
    np.testing.assert_array_equal(x.grad.data, expected)
    assert x.grad.shape == x.shape


@pytest.mark.parametrize('operation', [F.max, F.min])
def test_extrema_assign_full_gradient_to_each_tied_extremum(operation):
    x = Variable(np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]))
    y = operation(x, axis=1)
    y.grad = Variable(np.array([2.0, 3.0]))

    y.backward()

    np.testing.assert_array_equal(x.grad.data, [[2, 2, 2], [3, 3, 3]])


@pytest.mark.parametrize('operation', [F.max, F.min])
def test_extrema_scalar_backward(operation):
    x = Variable(np.array(2.0))
    operation(x).backward()

    np.testing.assert_array_equal(x.grad.data, 1)


def test_clip_forward_and_backward_include_boundary_values():
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Variable(data.copy())
    y = F.clip(x, -1.0, 1.0)
    y.grad = Variable(np.array([1, 2, 3, 4, 5], dtype=np.float32))

    y.backward()

    np.testing.assert_array_equal(y.data, [-1, -1, 0, 1, 1])
    np.testing.assert_array_equal(x.grad.data, [0, 2, 3, 4, 0])
    np.testing.assert_array_equal(x.data, data)
    assert y.dtype == np.float32 and x.grad.dtype == np.float32


def test_clip_preserves_higher_order_gradients_inside_bounds():
    x = Variable(np.array([-2.0, -0.5, 0.5, 2.0]))
    (F.clip(x, -1.0, 1.0) ** 2).sum().backward(create_graph=True)
    first_grad = x.grad
    np.testing.assert_allclose(first_grad.data, [0, -1, 1, 0])
    x.cleargrad()

    first_grad.sum().backward()

    np.testing.assert_allclose(x.grad.data, [0, 2, 2, 0])
