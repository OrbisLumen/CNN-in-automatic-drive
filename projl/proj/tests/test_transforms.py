"""Transform order, channel broadcasting, dtype, and input preservation."""

import numpy as np
import pytest

from mydezero.transforms import AsType, Compose, Flatten, Normalize, ToFloat, ToInt


def test_compose_identity_and_order():
    x = np.array([1, 2])
    assert Compose()(x) is x
    np.testing.assert_array_equal(Compose([lambda a: a + 1, lambda a: a * 3])(x), [6, 9])


def test_mnist_transform_preserves_raw_pixels():
    x = np.array([[[0, 255], [127, 64]]], dtype=np.uint8)
    original = x.copy()
    result = Compose([Flatten(), ToFloat(), Normalize(0, 255)])(x)
    assert result.shape == (4,) and result.dtype == np.float32
    np.testing.assert_allclose(result, [0, 1, 127 / 255, 64 / 255])
    np.testing.assert_array_equal(x, original)


def test_normalize_channel_first_values():
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    result = Normalize([1, 4], [2, 3])(x)
    np.testing.assert_allclose(result[0], (x[0] - 1) / 2)
    np.testing.assert_allclose(result[1], (x[1] - 4) / 3)
    assert result.dtype == np.float32


@pytest.mark.parametrize('transform,dtype', [
    (AsType(), np.float32), (ToFloat(np.float64), np.float64),
    (ToInt(), np.dtype(int)), (ToInt(np.int32), np.int32),
])
def test_dtype_transforms(transform, dtype):
    x = np.array([1.0, 2.0])
    result = transform(x)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, x)
