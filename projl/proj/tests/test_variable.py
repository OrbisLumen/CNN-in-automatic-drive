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


def test_repr():
    x = Variable(np.array([[1, 2]]))
    assert repr(x) == 'variable([[1 2]])'


def test_add_and_mul_reload():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))
    y = a * b + c
    assert y.data == 7.0


def test_add_and_mul_with_float():
    x = Variable(np.array(2.0))
    y = 3.0 * x + 1.0
    assert y.data == 7.0


def test_add_and_mul_with_ndarray():
    x = Variable(np.array(1.0))
    y = np.array([2.0]) + x
    assert y.data == 3.0


def test_neg():
    x = Variable(np.array(-2.0))
    y = -x
    assert y.data == 2.0


def test_sub_and_rsub():
    x = Variable(np.array(-2.0))
    assert (x - 1.0).data == -3.0
    assert (1.0 - x).data == 3.0


def test_div_and_rdiv():
    x = Variable(np.array(2.0))
    y = Variable(np.array(1.0))

    assert (x / 2).data == 1.0

    z = x / y
    z.backward()
    assert x.grad.data == 1.0
    assert y.grad.data == -2.0
    assert z.data == 2.0


def test_pow():
    x = Variable(np.array(2.0))
    y = x ** 3
    y.backward()
    assert y.data == 8.0
    assert x.grad.data == 12.0
