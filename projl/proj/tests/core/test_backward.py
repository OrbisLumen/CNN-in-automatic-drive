"""Graph traversal, gradient accumulation, and higher-order derivatives."""

import numpy as np
import pytest

from mydezero import Variable
from mydezero.core import add, mul
from mydezero.functions import exp, square
from tests.helpers import numerical_diff


def test_composed_function_backward():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()

    np.testing.assert_allclose(x.grad.data, 3.297442541400256)


def test_backward_accepts_ndarray_output_gradient():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.grad = np.array(1.0)

    y.backward()

    np.testing.assert_allclose(x.grad.data, 3.297442541400256)


def test_square_backward_matches_numerical_gradient():
    x = Variable(np.array(0.37))
    y = square(x)

    y.backward()

    np.testing.assert_allclose(x.grad.data, numerical_diff(square, x))


def test_add_and_square_backward():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = add(square(x0), square(x1))

    y.backward()

    np.testing.assert_allclose(y.data, 13.0)
    np.testing.assert_allclose(x0.grad.data, 4.0)
    np.testing.assert_allclose(x1.grad.data, 6.0)


@pytest.mark.parametrize('uses', [2, 3], ids=['two-uses', 'three-uses'])
def test_reused_variable_accumulates_gradient(uses):
    x = Variable(np.array(3.0))
    y = add(x, x)
    if uses == 3:
        y = add(y, x)

    y.backward()

    np.testing.assert_allclose(x.grad.data, uses)


def test_branching_graph_backward():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))

    y.backward()

    np.testing.assert_allclose(y.data, 32.0)
    np.testing.assert_allclose(x.grad.data, 64.0)


def test_backward_discards_intermediate_gradients_by_default():
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    intermediate = add(x0, x1)
    y = add(x0, intermediate)

    y.backward()

    assert y.grad is None
    assert intermediate.grad is None
    np.testing.assert_allclose(x0.grad.data, 2.0)
    np.testing.assert_allclose(x1.grad.data, 1.0)


def test_mul_function_backward():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = add(mul(a, b), c)
    y.backward()

    np.testing.assert_allclose(a.grad.data, 2.0)
    np.testing.assert_allclose(b.grad.data, 3.0)


def test_second_backward_accumulates_existing_gradient():
    x = Variable(np.array(2.0))
    y = x ** 4 - 2 * x ** 2
    y.backward(create_graph=True)
    np.testing.assert_allclose(x.grad.data, 24.0)

    gx = x.grad
    gx.backward()

    # The first gradient remains: 24 + (12 * x**2 - 4) = 68 at x = 2.
    np.testing.assert_allclose(x.grad.data, 68.0)
