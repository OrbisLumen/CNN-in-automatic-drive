"""Optimization example functions and their gradients."""

import numpy as np
import pytest

from mydezero import Variable
from mydezero.functions import goldstein, matyas, sphere


@pytest.mark.parametrize(
    'function, expected_gradients',
    [
        pytest.param(sphere, (2.0, 2.0), id='sphere'),
        pytest.param(matyas, (0.040000000000000036, 0.040000000000000036), id='matyas'),
        pytest.param(goldstein, (-5376.0, 8064.0), id='goldstein'),
    ],
)
def test_optimization_function_gradients(function, expected_gradients):
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))

    function(x, y).backward()

    np.testing.assert_allclose(x.grad.data, expected_gradients[0])
    np.testing.assert_allclose(y.grad.data, expected_gradients[1])
