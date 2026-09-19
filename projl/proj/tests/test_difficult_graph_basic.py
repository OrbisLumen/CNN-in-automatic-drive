import numpy as np
from mydezero import Variable, square, add

def test_basic_difficult_graph():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    assert y.data == 32.0
    assert x.grad == 64.0

if __name__ == "__main__":
    test_basic_difficult_graph()