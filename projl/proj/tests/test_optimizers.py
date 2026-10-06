"""SGD and momentum updates, gradient filtering, and optimizer hooks."""

import numpy as np
import pytest

from mydezero import Parameter, Variable
from mydezero.optimizers import MomentumSGD, SGD


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


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_momentum_sgd_persists_velocity_across_changing_gradients(dtype):
    param = Parameter(np.array([1.0, -2.0], dtype=dtype))
    optimizer = MomentumSGD(lr=0.1, momentum=0.5)
    param.grad = Variable(np.array([2.0, -4.0], dtype=dtype))

    optimizer.update_one(param)
    np.testing.assert_allclose(param.data, [0.8, -1.6])
    np.testing.assert_allclose(optimizer.vs[id(param)], [-0.2, 0.4])
    param.grad = Variable(np.array([-1.0, 2.0], dtype=dtype))
    optimizer.update_one(param)
    np.testing.assert_allclose(param.data, [0.8, -1.6])
    np.testing.assert_allclose(optimizer.vs[id(param)], [0.0, 0.0], atol=1e-7)
    param.grad = Variable(np.zeros(2, dtype=dtype))
    optimizer.update_one(param)
    assert optimizer.vs[id(param)].dtype == dtype


def test_momentum_sgd_filters_gradients_and_keeps_independent_velocities(regression_model):
    model = regression_model
    optimizer = MomentumSGD(lr=0.1, momentum=0.5).setup(model)
    hook_calls = []

    def scale_gradients(params):
        hook_calls.append(set(params))
        for param in params:
            param.grad.data *= 0.5

    optimizer.add_hook(scale_gradients)
    model.linear.W.grad = Variable(np.array([[-4.0]]))
    model.linear.b.grad = Variable(np.array([-2.0]))
    optimizer.update()
    np.testing.assert_allclose(model.linear.W.data, [[0.2]])
    np.testing.assert_allclose(model.linear.b.data, [0.1])
    model.cleargrads()
    model.linear.W.grad = Variable(np.zeros((1, 1)))
    optimizer.update()

    assert optimizer.target is model
    assert hook_calls == [{model.linear.W, model.linear.b}, {model.linear.W}]
    np.testing.assert_allclose(model.linear.W.data, [[0.3]])
    np.testing.assert_allclose(model.linear.b.data, [0.1])
    np.testing.assert_allclose(optimizer.vs[id(model.linear.W)], [[0.1]])
    np.testing.assert_allclose(optimizer.vs[id(model.linear.b)], [0.1])
    assert id(model.unused) not in optimizer.vs
    np.testing.assert_allclose(model.unused.data, 7.0)


def test_zero_momentum_matches_sgd_over_multiple_updates():
    plain = Parameter(np.array([1.0, -2.0]))
    momentum = Parameter(plain.data.copy())
    sgd = SGD(lr=0.1)
    optimizer = MomentumSGD(lr=0.1, momentum=0)

    for gradient in ([2.0, -4.0], [-1.0, 2.0], [0.0, 3.0]):
        plain.grad = Variable(np.array(gradient))
        momentum.grad = Variable(np.array(gradient))
        sgd.update_one(plain)
        optimizer.update_one(momentum)
        np.testing.assert_allclose(momentum.data, plain.data)
