"""SGD updates, gradient filtering, and optimizer hooks."""

import numpy as np

from mydezero import Variable
from mydezero.optimizers import SGD


def test_sgd_updates_only_parameters_with_gradients_and_runs_hooks(regression_model):
    model = regression_model
    model.linear.W.grad = Variable(np.array([[-10.0]]))
    model.linear.b.grad = Variable(np.array([-6.0]))
    optimizer = SGD(lr=0.1).setup(model)
    hook_calls = []

    def scale_gradients(params):
        hook_calls.append(set(params))
        for param in params:
            param.grad.data *= 0.5

    optimizer.add_hook(scale_gradients)
    optimizer.update()

    assert optimizer.target is model
    assert hook_calls == [{model.linear.W, model.linear.b}]
    np.testing.assert_allclose(model.linear.W.data, [[0.5]])
    np.testing.assert_allclose(model.linear.b.data, [0.3])
    np.testing.assert_allclose(model.unused.data, 7.0)
