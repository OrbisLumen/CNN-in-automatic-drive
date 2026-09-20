import numpy as np
from mydezero import Variable
from mydezero.functions import sphere, matyas, goldstein


def test_several_test_functions_for_optimization():
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z1 = sphere(x, y)
    z1.backward()
    assert x.grad == 2.0
    assert y.grad == 2.0

    x.cleargrad()
    y.cleargrad()
    z2 = matyas(x, y)
    z2.backward()
    np.testing.assert_allclose(x.grad, 0.040000000000000036)
    np.testing.assert_allclose(y.grad, 0.040000000000000036)

    x.cleargrad()
    y.cleargrad()
    z3 = goldstein(x, y)
    z3.backward()
    assert x.grad == -5376.0
    assert y.grad == 8064.0
