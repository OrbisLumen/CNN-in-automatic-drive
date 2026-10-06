"""A complete model, loss, backward, and optimizer training step."""

import numpy as np

from mydezero import Variable
import mydezero.functions as F
from mydezero.optimizers import SGD


def test_nested_model_training_step_and_optimizer_hooks(regression_model):
    model = regression_model
    x = Variable(np.array([[1.0], [2.0]]))
    target = Variable(np.array([[2.0], [4.0]]))
    optimizer = SGD(lr=0.1).setup(model)

    def scale_gradients(params):
        for param in params:
            param.grad.data *= 0.5

    optimizer.add_hook(scale_gradients)
    loss = F.mean_squared_error(model(x), target)
    loss.backward()
    optimizer.update()

    np.testing.assert_allclose(model.linear.W.data, [[0.5]])
    np.testing.assert_allclose(model.linear.b.data, [0.3])
    np.testing.assert_allclose(model.unused.data, 7.0)
    assert F.mean_squared_error(model(x), target).data < loss.data
    model.cleargrads()
    assert all(param.grad is None for param in model.params())
