import numpy as np
import pytest

from mydezero import Model, Parameter, Variable
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.optimizers import SGD


@pytest.mark.parametrize('nobias', [False, True])
def test_linear_lazy_initialization_and_gradients(nobias):
    layer = Linear(2, nobias=nobias, dtype=np.float64)
    assert layer.W.data is None
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    y = layer(x)
    weights = layer.W.data.copy()
    y.sum().backward()

    expected = x.data @ weights
    if not nobias:
        expected += layer.b.data
        np.testing.assert_allclose(layer.b.grad.data, [2.0, 2.0])
    else:
        assert layer.b is None
    np.testing.assert_allclose(y.data, expected)
    np.testing.assert_allclose(x.grad.data, np.ones((2, 2)) @ weights.T)
    np.testing.assert_allclose(layer.W.grad.data, x.data.T @ np.ones((2, 2)))
    assert layer.W.data is not None
    assert layer.W.data.dtype == np.float64
    layer(x)
    np.testing.assert_array_equal(layer.W.data, weights)


def test_nested_model_training_step_and_optimizer_hooks():
    class Regression(Model):
        def __init__(self):
            super().__init__()
            self.linear = Linear(1, dtype=np.float64, in_size=1)
            self.linear.W.data[:] = 0
            self.unused = Parameter(np.array(7.0))

        def forward(self, x):
            return self.linear(x)

    model = Regression()
    x = Variable(np.array([[1.0], [2.0]]))
    target = Variable(np.array([[2.0], [4.0]]))
    optimizer = SGD(lr=0.1).setup(model)
    hook_calls = []

    def scale_gradients(params):
        hook_calls.append(set(params))
        for param in params:
            param.grad.data *= 0.5

    optimizer.add_hook(scale_gradients)
    loss = F.mean_squared_error(model(x), target)
    loss.backward()
    optimizer.update()

    assert optimizer.target is model
    assert set(model.params()) == {model.linear.W, model.linear.b, model.unused}
    assert hook_calls == [{model.linear.W, model.linear.b}]
    np.testing.assert_allclose(model.linear.W.data, [[0.5]])
    np.testing.assert_allclose(model.linear.b.data, [0.3])
    np.testing.assert_allclose(model.unused.data, 7.0)
    assert F.mean_squared_error(model(x), target).data < loss.data
    model.cleargrads()
    assert all(param.grad is None for param in model.params())
