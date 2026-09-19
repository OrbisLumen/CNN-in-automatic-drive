import pytest
import numpy as np
from mydezero import *
from tests.helper import numerical_diff


def test_square_basic():
    x = Variable(np.array(10))
    y = square(x)
    assert y.data == np.array(100)


def test_consistent_call():
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)
    np.testing.assert_allclose(y.data,
                               1.648721270700128,
                               rtol=1e-5,
                               atol=1e-7,
                               )


def test_basic_backward():

    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    np.testing.assert_allclose(x.grad,
                               3.297442541400256,
                               rtol=1e-5,
                               atol=1e-7, )

def test_basic_backward_with_initialization():

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    np.testing.assert_allclose(x.grad,
                               3.297442541400256,
                               rtol=1e-5,
                               atol=1e-7, )

def test_basic_backward_with_numerical_diff():
    x = Variable(np.random.rand(1))  # 笠成随机的输入僵
    y = square(x)
    y.backward()
    num_grad = numerical_diff(square, x)
    flg = np.allclose(x.grad, num_grad)
    assert flg

if __name__ == "__main__":
    test_square_basic()
    test_consistent_call()
    test_basic_backward()
    test_basic_backward_with_initialization()
    test_basic_backward_with_numerical_diff()
