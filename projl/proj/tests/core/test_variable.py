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


def test_conversion_helpers_preserve_arrays_and_variables():
    from mydezero.core import as_array, as_variable

    data = np.array([1., 2.])
    x = Variable(data)
    assert as_array(data) is data
    assert as_variable(x) is x
    assert as_variable(data).data is data
    scalar = as_array(3.)
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()
    assert Variable(None).data is None
    assert len(x) == 2
