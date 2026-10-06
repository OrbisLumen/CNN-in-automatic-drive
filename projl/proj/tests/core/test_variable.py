"""Variable metadata, gradient clearing, and convenience methods."""

import numpy as np
import pytest

from mydezero import Variable


def test_variable_exposes_properties():
    x = Variable(np.array([[1.0, 2.0]]))

    assert x.shape == (1, 2)
    assert x.dtype == np.float64
    assert x.ndim == 2
    assert x.size == 2


def test_variable_rejects_non_array_data():
    with pytest.raises(TypeError):
        Variable(1.0)


def test_cleargrad_removes_accumulated_gradient():
    x = Variable(np.array(2.0))
    x.grad = np.array(3.0)

    x.cleargrad()

    assert x.grad is None


def test_variable_repr():
    x = Variable(np.array([[1, 2]]))
    assert repr(x) == 'variable([[1 2]])'


def test_variable_reshape_backward_restores_input_shape():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.reshape((6,))
    y.backward(retain_grad=True)

    np.testing.assert_array_equal(x.grad.data, np.ones_like(x.data))
