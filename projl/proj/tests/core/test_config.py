"""Backpropagation configuration, test mode, and context restoration."""

import numpy as np

from mydezero import Variable, using_config
from mydezero.functions import square


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


def test_nested_context_and_exception_restore_configuration():
    from mydezero import Config, no_grad

    original = Config.enable_backprop
    try:
        with no_grad():
            assert Config.enable_backprop is False
            with using_config('enable_backprop', True):
                assert Config.enable_backprop is True
            assert Config.enable_backprop is False
            raise RuntimeError('test context cleanup')
    except RuntimeError:
        pass
    assert Config.enable_backprop == original


def test_test_mode_passes_dropout_input_through_unchanged():
    from mydezero import test_mode
    from mydezero.functions import dropout

    x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with test_mode():
        y = dropout(x, 0.9)

    np.testing.assert_array_equal(y.data, x.data)


def test_test_mode_restores_train_flag():
    from mydezero import Config, test_mode

    with test_mode():
        assert Config.train is False
    assert Config.train is True
