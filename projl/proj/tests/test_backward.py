import numpy as np

from mydezero import Variable, add, exp, square, mul
from tests.helpers import numerical_diff


def test_composed_function_backward():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()

    np.testing.assert_allclose(x.grad, 3.297442541400256)


def test_backward_respects_initial_output_gradient():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.grad = np.array(1.0)

    y.backward()

    np.testing.assert_allclose(x.grad, 3.297442541400256)


def test_square_backward_matches_numerical_gradient():
    x = Variable(np.array(0.37))
    y = square(x)

    y.backward()

    np.testing.assert_allclose(x.grad, numerical_diff(square, x))


def test_add_and_square_backward():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = add(square(x0), square(x1))

    y.backward()

    np.testing.assert_allclose(y.data, 13.0)
    np.testing.assert_allclose(x0.grad, 4.0)
    np.testing.assert_allclose(x1.grad, 6.0)


def test_reused_variable_accumulates_gradient():
    x = Variable(np.array(3.0))
    y = add(x, x)

    y.backward()

    np.testing.assert_allclose(x.grad, 2.0)


def test_reused_variable_accumulates_gradient_across_multiple_adds():
    x = Variable(np.array(3.0))
    y = add(add(x, x), x)

    y.backward()

    np.testing.assert_allclose(x.grad, 3.0)


def test_branching_graph_backward():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))

    y.backward()

    np.testing.assert_allclose(y.data, 32.0)
    np.testing.assert_allclose(x.grad, 64.0)


def test_backward_discards_intermediate_gradients_by_default():
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    intermediate = add(x0, x1)
    y = add(x0, intermediate)

    y.backward()

    assert y.grad is None
    assert intermediate.grad is None
    np.testing.assert_allclose(x0.grad, 2.0)
    np.testing.assert_allclose(x1.grad, 1.0)

def test_mul_function_backward():

    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = add(mul(a, b), c)
    y.backward()

    np.testing.assert_allclose(a.grad, 2.0)
    np.testing.assert_allclose(b.grad, 3.0)
