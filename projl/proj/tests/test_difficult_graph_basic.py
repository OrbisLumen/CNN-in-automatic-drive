import numpy as np
from mydezero import *

def test_basic_difficult_graph():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    assert y.data == 32.0
    assert x.grad == 64.0

def test_not_retain_grad():
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    assert y.grad == None
    assert t.grad == None

    assert x0.grad == 2.0
    assert x1.grad == 1.0

def test_no_grad():
    def no_grad():
        return using_config('enable_backprop', False)

    x = Variable(np.array(2.0))
    y = square(x)

    assert y.data == 4.0
    assert x.grad == None

if __name__ == "__main__":
    test_basic_difficult_graph()
    test_not_retain_grad()
    test_no_grad()