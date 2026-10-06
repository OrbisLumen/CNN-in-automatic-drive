"""Stable softmax and batch-averaged classification loss gradients."""

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


@pytest.mark.parametrize('axis', [0, 1, -1, None, (0, 2)])
def test_softmax_normalizes_selected_axes_without_mutating_input(axis):
    data = np.arange(24.0).reshape(2, 3, 4) - 12
    original = data.copy()

    probabilities = F.softmax(data, axis=axis)

    np.testing.assert_allclose(probabilities.data.sum(axis=axis), 1)
    assert probabilities.shape == data.shape
    assert np.all(probabilities.data > 0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_softmax_backward_matches_numerical_gradient(axis):
    data = np.array([[0.2, -0.5, 1.3], [1.0, 0.4, -0.8]])
    weights = np.array([[0.5, 2.0, -1.0], [-0.4, 1.5, 0.7]])
    x = Variable(data.copy())

    def objective(values):
        return (F.softmax(values, axis=axis) * weights).sum()

    objective(x).backward()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, data), atol=1e-8)
    np.testing.assert_allclose(x.grad.data.sum(axis=axis), 0, atol=1e-15)


def test_softmax_large_logits_do_not_overflow():
    data = np.array([[1000.0, 1000.0, 1000.0], [1000.0, -1000.0, 0.0]])

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        probabilities = F.softmax(data)

    np.testing.assert_allclose(probabilities.data, [[1 / 3, 1 / 3, 1 / 3], [1, 0, 0]])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('column_labels', [False, True], ids=['vector', 'column'])
def test_cross_entropy_forward_backward_shape_dtype_and_label_gradients(dtype, column_labels):
    probabilities = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=dtype)
    data = np.log(probabilities)
    x = Variable(data.copy())
    labels = np.array([2, 0], dtype=np.int64)
    t = Variable(labels[:, None] if column_labels else labels.copy())

    loss = F.softmax_cross_entropy_loss(x, t)
    loss.backward()

    np.testing.assert_allclose(loss.data, -np.log(0.5 * 0.6) / 2, rtol=1e-6)
    np.testing.assert_allclose(x.grad.data, [[0.1, 0.15, -0.25], [-0.2, 0.05, 0.15]], rtol=1e-6)
    assert loss.shape == () and loss.dtype == dtype
    assert x.grad.shape == x.shape and x.grad.dtype == dtype
    assert t.grad is None
    np.testing.assert_array_equal(x.data, data)
    np.testing.assert_array_equal(t.data.ravel(), labels)


@pytest.mark.parametrize('labels', [np.array([2, 0], dtype=np.int32), np.array([[2], [0]])])
def test_scaled_cross_entropy_matches_numerical_gradient(labels):
    data = np.array([[0.2, -0.5, 1.3], [1.0, 0.4, -0.8]])
    x = Variable(data.copy())

    def objective(values):
        return F.softmax_cross_entropy_loss(values, labels) * 3.5

    objective(x).backward()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, data), atol=1e-8)


def test_cross_entropy_supports_second_order_directional_derivatives():
    probabilities = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    x = Variable(np.log(probabilities))
    direction = np.array([[1.0, -2.0, 0.5], [-0.5, 1.0, 2.0]])
    F.softmax_cross_entropy_loss(x, np.array([2, 0])).backward(create_graph=True)
    first_grad = x.grad
    x.cleargrad()

    (first_grad * direction).sum().backward()

    expected = probabilities * (
        direction - (probabilities * direction).sum(axis=1, keepdims=True)
    ) / 2
    np.testing.assert_allclose(x.grad.data, expected, atol=1e-15)


@pytest.mark.parametrize('dtype, offset', [(np.float32, 1e20), (np.float64, 1e16)])
def test_cross_entropy_retains_accuracy_for_large_identical_logits(dtype, offset):
    x = Variable(np.full((2, 3), offset, dtype=dtype))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        loss = F.softmax_cross_entropy_loss(x, np.array([0, 2]))
        loss.backward()

    np.testing.assert_allclose(loss.data, np.log(3), rtol=1e-6)
    assert np.all(np.isfinite(x.grad.data))


def test_cross_entropy_handles_underflowed_probabilities():
    x = Variable(np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]]))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        loss = F.softmax_cross_entropy_loss(x, np.array([1, 1]))
        loss.backward()

    np.testing.assert_allclose(loss.data, 1000.0)
    np.testing.assert_allclose(x.grad.data, [[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]])


def test_single_class_cross_entropy_is_zero():
    x = Variable(np.array([[3.0], [-4.0]]))
    loss = F.softmax_cross_entropy_loss(x, np.array([0, 0]))
    loss.backward()

    np.testing.assert_allclose(loss.data, 0)
    np.testing.assert_allclose(x.grad.data, np.zeros_like(x.data))


@pytest.mark.parametrize(
    'logits, labels, error',
    [
        pytest.param(np.zeros(3), np.array([0]), ValueError, id='one-dimensional-logits'),
        pytest.param(np.empty((0, 3)), np.array([], dtype=int), ValueError, id='empty-batch'),
        pytest.param(np.empty((2, 0)), np.array([0, 0]), ValueError, id='empty-classes'),
        pytest.param(np.zeros((2, 3)), np.array([0]), ValueError, id='wrong-label-count'),
        pytest.param(np.zeros((2, 3)), np.array([[0, 1]]), ValueError, id='wrong-label-shape'),
        pytest.param(np.zeros((2, 3)), np.eye(3)[:2], ValueError, id='one-hot-labels'),
        pytest.param(np.zeros((2, 3)), np.array([0.0, 1.0]), TypeError, id='float-labels'),
        pytest.param(np.zeros((2, 3)), np.array([True, False]), TypeError, id='bool-labels'),
        pytest.param(np.zeros((2, 3)), np.array([-1, 0]), ValueError, id='negative-label'),
        pytest.param(np.zeros((2, 3)), np.array([0, 3]), ValueError, id='out-of-range-label'),
    ],
)
def test_cross_entropy_rejects_invalid_inputs(logits, labels, error):
    with pytest.raises(error):
        F.softmax_cross_entropy_loss(logits, labels)
