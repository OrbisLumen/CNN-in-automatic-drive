import numpy as np
import pytest

from mydezero import Variable


def test_variable_exposes_data_shape():
    x = Variable(np.array([[1.0, 2.0]]))

    assert x.shape == (1, 2)


def test_variable_rejects_non_array_data():
    with pytest.raises(TypeError):
        Variable(1.0)


def test_cleargrad_removes_accumulated_gradient():
    x = Variable(np.array(2.0))
    x.grad = np.array(3.0)

    x.cleargrad()

    assert x.grad is None
