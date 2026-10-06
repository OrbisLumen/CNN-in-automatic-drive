"""A complete model, loss, backward, and optimizer training step."""

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.models import MLP, Sequential
from mydezero.optimizers import MomentumSGD, SGD


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


@pytest.mark.parametrize('kind', ['sequential', 'mlp'])
def test_composed_model_training_with_momentum_reduces_loss(kind):
    model = Sequential(Linear(2), Linear(1)) if kind == 'sequential' else MLP((2, 1))
    x = Variable(np.array([[0.0], [1.0], [2.0]]))
    target = Variable(np.array([[1.0], [3.0], [5.0]]))
    model(x)
    for layer in model.layers:
        layer.W.data[:] = 0.1
        layer.b.data[:] = 0
    optimizer = MomentumSGD(lr=0.02, momentum=0.5).setup(model)
    initial_loss = float(F.mean_squared_error(model(x), target).data)

    for _ in range(30):
        model.cleargrads()
        loss = F.mean_squared_error(model(x), target)
        loss.backward()
        assert all(param.grad is not None for param in model.params())
        optimizer.update()

    final_loss = float(F.mean_squared_error(model(x), target).data)
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 0.5


def test_softmax_cross_entropy_trains_a_classifier_with_momentum():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    model = MLP((2,))
    model(x)
    model.l0.W.data = np.zeros((1, 2), dtype=np.float32)
    model.l0.b.data[:] = 0
    optimizer = MomentumSGD(lr=0.1, momentum=0.5).setup(model)
    initial_loss = float(F.softmax_cross_entropy_loss(model(x), labels).data)

    for _ in range(30):
        model.cleargrads()
        loss = F.softmax_cross_entropy_loss(model(x), labels)
        loss.backward()
        for param in model.params():
            assert param.grad.shape == param.shape
            assert param.grad.dtype == param.dtype
        optimizer.update()

    logits = model(x)
    assert float(F.softmax_cross_entropy_loss(logits, labels).data) < initial_loss * 0.25
    np.testing.assert_array_equal(logits.data.argmax(axis=1), labels)
