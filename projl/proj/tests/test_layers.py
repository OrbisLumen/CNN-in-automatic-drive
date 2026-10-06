"""Linear layer initialization, forward output, and gradients."""

import numpy as np
import pytest

from mydezero import Variable
from mydezero.layers import Linear


@pytest.mark.parametrize('nobias', [False, True])
def test_linear_lazy_initialization_and_gradients(nobias):
    layer = Linear(2, nobias=nobias, dtype=np.float64)
    assert layer.W.data is None
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    y = layer(x)
    weights = layer.W.data.copy()
    y.sum().backward()

    expected = x.data @ weights
    if not nobias:
        expected += layer.b.data
        np.testing.assert_allclose(layer.b.grad.data, [2.0, 2.0])
    else:
        assert layer.b is None
    np.testing.assert_allclose(y.data, expected)
    np.testing.assert_allclose(x.grad.data, np.ones((2, 2)) @ weights.T)
    np.testing.assert_allclose(layer.W.grad.data, x.data.T @ np.ones((2, 2)))
    assert layer.W.data is not None
    assert layer.W.data.dtype == np.float64
    layer(x)
    np.testing.assert_array_equal(layer.W.data, weights)
