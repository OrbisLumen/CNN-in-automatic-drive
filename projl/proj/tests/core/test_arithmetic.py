"""Arithmetic functions and Variable operator overloads."""

import numpy as np
import pytest

from mydezero import Variable
from mydezero.core import add


def test_add_and_mul_operator_overloads():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = a * b + c

    np.testing.assert_allclose(y.data, 7.0)


def test_add_and_mul_with_float():
    x = Variable(np.array(2.0))

    y = 3.0 * x + 1.0

    np.testing.assert_allclose(y.data, 7.0)


def test_ndarray_left_addition():
    x = Variable(np.array(1.0))

    y = np.array([2.0]) + x

    np.testing.assert_allclose(y.data, [3.0])


def test_negation_forward():
    x = Variable(np.array(-2.0))

    y = -x

    np.testing.assert_allclose(y.data, 2.0)


@pytest.mark.parametrize(
    'reverse, expected',
    [pytest.param(False, -3.0, id='subtract'), pytest.param(True, 3.0, id='reverse-subtract')],
)
def test_subtraction_operator_overloads(reverse, expected):
    x = Variable(np.array(-2.0))

    y = 1.0 - x if reverse else x - 1.0

    np.testing.assert_allclose(y.data, expected)


def test_division_by_scalar_forward():
    x = Variable(np.array(2.0))

    np.testing.assert_allclose((x / 2).data, 1.0)


def test_division_forward_and_backward():
    x = Variable(np.array(2.0))
    y = Variable(np.array(1.0))

    z = x / y
    z.backward()
    np.testing.assert_allclose(z.data, 2.0)
    np.testing.assert_allclose(x.grad.data, 1.0)
    np.testing.assert_allclose(y.grad.data, -2.0)


def test_power_forward_and_backward():
    x = Variable(np.array(2.0))
    y = x ** 3
    y.backward()
    np.testing.assert_allclose(y.data, 8.0)
    np.testing.assert_allclose(x.grad.data, 12.0)


def test_tensor_addition_forward():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    y = x + c
    expected = np.array([[11, 22, 33], [44, 55, 66]])

    np.testing.assert_array_equal(y.data, expected)


def test_add_forward():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))

    y = add(x0, x1)

    np.testing.assert_allclose(y.data, np.array(5.0))
