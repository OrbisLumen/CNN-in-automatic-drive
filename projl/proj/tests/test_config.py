import numpy as np

from mydezero import Variable, using_config
from mydezero.functions import exp, square


def test_disabling_backprop_does_not_create_graph():
    x = Variable(np.array(2.0))

    with using_config("enable_backprop", False):
        y = square(x)

    np.testing.assert_allclose(y.data, 4.0)
    assert y.creator is None


def test_backprop_is_restored_after_context():
    x = Variable(np.array(2.0))

    with using_config("enable_backprop", False):
        square(x)

    y = square(x)

    assert y.creator is not None
