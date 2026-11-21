"""DOT graph generation and numerical reduction helpers."""

import numpy as np
import pytest

from mydezero import Variable, no_grad
from mydezero.utils import get_dot_graph, logsumexp, max_backward_shape


def test_get_dot_graph_for_leaf_variable():
    x = Variable(np.array(1.0), name='x')
    graph = get_dot_graph(x)

    assert graph.startswith('digraph G {')
    assert 'label="x: () float64"' in graph
    assert '->' not in graph


@pytest.mark.parametrize('axis', [None, 0, 1, -1, (0, 2)])
def test_logsumexp_matches_logaddexp_and_preserves_input(axis):
    data = np.arange(24.0).reshape(2, 3, 4) - 12
    original = data.copy()

    result = logsumexp(data, axis=axis)

    expected = np.logaddexp.reduce(data, axis=axis, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_logsumexp_avoids_overflow_and_retains_dtype(dtype):
    data = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]], dtype=dtype)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = logsumexp(data)

    np.testing.assert_allclose(result, [[1000 + np.log(3)], [-1000 + np.log(3)]], rtol=1e-6)
    assert result.shape == (2, 1) and result.dtype == dtype


@pytest.mark.parametrize(
    'axis, expected',
    [(None, (1, 1, 1)), (1, (2, 1, 4)), (-1, (2, 3, 1)),
     ((-3, -1), (1, 3, 1)), ((), (2, 3, 4)), (np.int64(-2), (2, 1, 4))],
)
def test_max_backward_shape_normalizes_reduction_axes(axis, expected):
    x = Variable(np.zeros((2, 3, 4)))

    assert max_backward_shape(x, axis) == expected


def test_get_dot_graph_without_backprop():
    with no_grad():
        y = Variable(np.array(1.0)) + 2

    graph = get_dot_graph(y)
    assert str(id(y)) in graph
    assert '->' not in graph
