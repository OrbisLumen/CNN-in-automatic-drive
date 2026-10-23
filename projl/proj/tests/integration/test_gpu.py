"""Real CuPy execution, transfers, CNN gradients, and optimizer updates."""

import numpy as np
import pytest

from mydezero import Variable, cuda
import mydezero.functions as F
from mydezero.models import MLP
from mydezero.optimizers import MomentumSGD


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f'CUDA device unavailable: {error}')
    if count == 0:
        pytest.skip('No CUDA device')
    # Compilation/library failures on an available GPU must fail, not skip.
    cupy.testing.assert_array_equal(cupy.arange(3) + 1, cupy.array([1, 2, 3]))
    return cupy


def test_device_transfer_and_broadcast_backward(cp):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = Variable(data.copy())
    x.to_gpu()
    assert isinstance(x.data, cp.ndarray)
    assert cuda.get_array_module(x) is cp
    bias = Variable(cp.array([1., 2., 3.], dtype=cp.float32))
    loss = ((x + bias) ** 2).sum()
    loss.backward()
    assert isinstance(x.grad.data, cp.ndarray)
    np.testing.assert_allclose(cuda.as_numpy(x.grad), 2 * (data + [1, 2, 3]))
    np.testing.assert_allclose(cuda.as_numpy(bias.grad), (2 * (data + [1, 2, 3])).sum(axis=0))
    x.to_cpu()
    assert cuda.get_array_module(x) is np
    np.testing.assert_array_equal(x.data, data)


@pytest.mark.parametrize('operation', [F.conv2d, F.deconv2d])
@pytest.mark.parametrize('stride,pad', [(1, 0), ((2, 1), (1, 0))])
def test_convolution_forward_and_all_gradients_match_cpu(cp, operation, stride, pad):
    rng = np.random.default_rng(7)
    weight_shape = (3, 2, 2, 3) if operation is F.conv2d else (2, 3, 2, 3)
    arrays = [rng.normal(size=(2, 2, 4, 5)), rng.normal(size=weight_shape), rng.normal(size=3)]
    cpu = [Variable(a.astype(np.float32)) for a in arrays]
    gpu = [Variable(cp.asarray(v.data)) for v in cpu]
    expected = operation(*cpu, stride=stride, pad=pad)
    actual = operation(*gpu, stride=stride, pad=pad)
    (expected ** 2).sum().backward()
    (actual ** 2).sum().backward()
    assert isinstance(actual.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual.data), expected.data, rtol=2e-5, atol=2e-5)
    for host, device in zip(cpu, gpu):
        assert isinstance(device.grad.data, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(device.grad.data), host.grad.data, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('operation', [F.pooling, F.average_pooling])
def test_pooling_forward_and_gradient_match_cpu(cp, operation):
    data = np.random.default_rng(8).normal(size=(2, 2, 4, 5)).astype(np.float32)
    cpu, gpu = Variable(data), Variable(cp.asarray(data))
    expected = operation(cpu, (2, 3), stride=(1, 2), pad=(1, 0))
    actual = operation(gpu, (2, 3), stride=(1, 2), pad=(1, 0))
    (expected ** 2).sum().backward()
    (actual ** 2).sum().backward()
    assert isinstance(actual.data, cp.ndarray)
    assert isinstance(gpu.grad.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual.data), expected.data, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gpu.grad.data), cpu.grad.data, atol=1e-6)


def test_gpu_classifier_training_and_model_transfer(cp):
    x = cp.array([[-2.], [-1.], [1.], [2.]], dtype=cp.float32)
    labels = cp.array([0, 0, 1, 1], dtype=cp.int64)
    model = MLP((2,))
    model.to_gpu()
    model(x)  # Lazy weights must also initialize on GPU.
    model.l0.W.data[:] = 0
    model.l0.b.data[:] = 0
    optimizer = MomentumSGD(lr=0.1, momentum=0.5).setup(model)
    initial = F.softmax_cross_entropy(model(x), labels).data.item()
    for _ in range(30):
        model.cleargrads()
        loss = F.softmax_cross_entropy(model(x), labels)
        loss.backward()
        for param in model.params():
            assert isinstance(param.data, cp.ndarray)
            assert isinstance(param.grad.data, cp.ndarray)
        optimizer.update()
    logits = model(x)
    assert F.softmax_cross_entropy(logits, labels).data.item() < initial * 0.25
    cp.testing.assert_array_equal(logits.data.argmax(axis=1), labels)
    expected = cp.asnumpy(logits.data)
    model.to_cpu()
    assert all(isinstance(p.data, np.ndarray) for p in model.params())
    np.testing.assert_allclose(model(cp.asnumpy(x)).data, expected, rtol=1e-5, atol=1e-6)
