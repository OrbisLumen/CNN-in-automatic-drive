import pytest
import numpy as np
from mydezero import Variable
from mydezero import add, square


def test_basic_add():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = add(x0, x1)
    assert y.data == 5


def test_basic_add_and_square():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))

    z = add(square(x0), square(x1))
    z.backward()
    assert z.data == 13.0
    assert x0.grad == 4.0
    assert x1.grad == 6.0


def test_same_variable_add():
    x = Variable(np.array(3))
    y = add(x, x)
    y.backward()
    assert x.grad == 2.0


def test_multi_add():
    x = Variable(np.array(3))
    z = add(add(x, x), x)
    z.backward()
    assert x.grad == 3.0


if __name__ == "__main__":
    test_basic_add()
    test_basic_add_and_square()
    test_same_variable_add()
    test_multi_add()
