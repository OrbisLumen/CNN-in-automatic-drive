"""Recursive parameter collection and gradient clearing on models."""

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.models import MLP, Sequential


def test_nested_model_collects_parameters(regression_model):
    model = regression_model

    assert set(model.params()) == {model.linear.W, model.linear.b, model.unused}


def test_nested_model_clears_all_gradients(regression_model):
    model = regression_model
    for param in model.params():
        param.grad = Variable(np.ones_like(param.data))

    model.cleargrads()

    assert all(param.grad is None for param in model.params())


def test_sequential_applies_layers_in_order_and_backpropagates():
    first = Linear(2, in_size=2, dtype=np.float64)
    second = Linear(1, in_size=2, dtype=np.float64)
    first.W.data[:] = [[1, 2], [3, 4]]
    first.b.data[:] = [1, -1]
    second.W.data[:] = [[2], [-1]]
    second.b.data[:] = [0.5]
    model = Sequential(first, second)
    x = Variable(np.array([[1.0, 2.0]]))

    y = model(x)
    y.sum().backward()

    np.testing.assert_allclose(y.data, [[7.5]])
    np.testing.assert_allclose(x.grad.data, [[0, 2]])
    np.testing.assert_allclose(first.W.grad.data, [[2, -1], [4, -2]])
    np.testing.assert_allclose(first.b.grad.data, [2, -1])
    np.testing.assert_allclose(second.W.grad.data, [[8], [9]])
    np.testing.assert_allclose(second.b.grad.data, [1])
    assert model.l0 is first and model.l1 is second
    assert set(model.params()) == {first.W, first.b, second.W, second.b}
    model.cleargrads()
    assert all(param.grad is None for param in model.params())


def test_empty_sequential_returns_input():
    model = Sequential()
    x = Variable(np.array([[1.0, 2.0]]))

    assert model(x) is x
    assert list(model.params()) == []


@pytest.mark.parametrize('activation', [None, F.tanh], ids=['default-sigmoid', 'tanh'])
def test_mlp_initializes_layers_and_activates_only_hidden_outputs(activation):
    model = MLP((3, 1)) if activation is None else MLP((3, 1), activation)
    assert all(layer.W.data is None for layer in model.layers)
    x = Variable(np.array([[1.0, -2.0], [0.5, 1.0]]))
    model(x)
    assert [layer.W.shape for layer in model.layers] == [(2, 3), (3, 1)]
    model.l0.W.data[:] = [[0.2, -0.3, 0.4], [0.5, 0.6, -0.7]]
    model.l0.b.data[:] = [0.1, -0.2, 0.3]
    model.l1.W.data[:] = [[1.0], [-2.0], [0.5]]
    model.l1.b.data[:] = [3.0]

    y = model(x)
    z = x.data @ model.l0.W.data + model.l0.b.data
    hidden = 1 / (1 + np.exp(-z)) if activation is None else np.tanh(z)
    expected = hidden @ model.l1.W.data + model.l1.b.data
    np.testing.assert_allclose(y.data, expected, rtol=1e-6)
    y.sum().backward()
    derivative = hidden * (1 - hidden) if activation is None else 1 - hidden ** 2
    hidden_grad = derivative * model.l1.W.data.T
    np.testing.assert_allclose(model.l0.W.grad.data, x.data.T @ hidden_grad, rtol=1e-6)
    np.testing.assert_allclose(model.l1.W.grad.data, hidden.sum(axis=0)[:, None])
    assert set(model.params()) == {model.l0.W, model.l0.b, model.l1.W, model.l1.b}
    model.cleargrads()
    assert all(param.grad is None for param in model.params())


def test_single_layer_mlp_does_not_apply_activation():
    def unexpected_activation(x):
        pytest.fail('A single-layer MLP has no hidden activation.')

    model = MLP((1,), activation=unexpected_activation)
    x = Variable(np.array([[2.0]]))
    model(x)
    model.l0.W.data[:] = 3
    model.l0.b.data[:] = -1

    np.testing.assert_allclose(model(x).data, [[5]])
