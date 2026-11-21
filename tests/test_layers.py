"""Linear layer initialization, forward output, gradients, and weight saving."""

import numpy as np
import pytest

from mydezero import Variable
from mydezero.layers import Linear, Conv2d, Deconv2d
from mydezero.models import MLP


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


@pytest.mark.parametrize('nobias', [False, True])
def test_save_load_weights_linear_roundtrip(nobias, tmp_path):
    layer = Linear(2, nobias=nobias, dtype=np.float64, in_size=3)
    path = str(tmp_path / 'linear.npz')
    layer.save_weights(path)

    restored = Linear(2, nobias=nobias, dtype=np.float64, in_size=3)
    restored.load_weights(path)

    np.testing.assert_array_equal(restored.W.data, layer.W.data)
    if not nobias:
        np.testing.assert_array_equal(restored.b.data, layer.b.data)
    else:
        assert restored.b is None


def test_save_load_weights_mlp_restores_predictions(tmp_path):
    model = MLP((4, 3))
    x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = model(x).data.copy()

    path = str(tmp_path / 'mlp.npz')
    model.save_weights(path)

    restored = MLP((4, 3))
    restored.load_weights(path)
    np.testing.assert_array_equal(restored(x).data, expected)


def test_conv2d_layer_lazy_initialization_and_shapes():
    layer = Conv2d(4, kernel_size=3, pad=1, dtype=np.float32)
    assert layer.W.data is None

    x = Variable(np.random.rand(2, 3, 8, 8).astype(np.float32))
    y = layer(x)

    assert y.shape == (2, 4, 8, 8)
    assert layer.W.shape == (4, 3, 3, 3)
    assert layer.b.shape == (4,)


def test_deconv2d_layer_forward_shape():
    layer = Deconv2d(4, kernel_size=3, pad=1, dtype=np.float32, in_channels=3)
    x = Variable(np.random.rand(2, 3, 5, 5).astype(np.float32))

    y = layer(x)

    assert y.shape == (2, 4, 5, 5)
