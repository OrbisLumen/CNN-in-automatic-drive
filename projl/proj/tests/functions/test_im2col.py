"""Patch extraction, overlap accumulation, and their derivatives."""

import numpy as np
import pytest

from mydezero import Variable, no_grad
import mydezero.functions as F
from mydezero.functions_conv import col2im_array, im2col_array


CASES = [
    pytest.param((2, 2, 3, 4), 2, 1, 0, id="overlap"),
    pytest.param((2, 2, 4, 5), (2, 3), (2, 1), (1, 2), id="rectangular-padding"),
    pytest.param((1, 1, 5, 7), 1, (2, 3), 0, id="uncovered-pixels"),
    pytest.param((1, 2, 3, 4), (3, 4), 1, 0, id="single-window"),
]


def reference_patches(data, kernel_size, stride, pad, to_matrix):
    """Extract each window directly, independently of the strided implementation."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    n, c, h, w = data.shape
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    patches = np.zeros((n, c, kh, kw, oh, ow), dtype=data.dtype)
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = oy * sh + ky - ph, ox * sw + kx - pw
                    if 0 <= iy < h and 0 <= ix < w:
                        patches[:, :, ky, kx, oy, ox] = data[:, :, iy, ix]
    if to_matrix:
        return np.stack([
            patches[batch, :, :, :, oy, ox].ravel()
            for batch in range(n) for oy in range(oh) for ox in range(ow)
        ])
    return patches


@pytest.mark.parametrize("shape,kernel_size,stride,pad", CASES)
@pytest.mark.parametrize("to_matrix", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_im2col_extracts_expected_patches(shape, kernel_size, stride, pad, to_matrix, dtype):
    data = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    original = data.copy()
    expected = reference_patches(data, kernel_size, stride, pad, to_matrix)

    actual = im2col_array(data, kernel_size, stride, pad, to_matrix)
    wrapped = F.im2col(data, kernel_size, stride, pad, to_matrix)

    assert isinstance(wrapped, Variable)
    assert actual.dtype == wrapped.dtype == data.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(wrapped.data, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("to_matrix", [False, True])
def test_im2col_handles_noncontiguous_input(to_matrix):
    data = np.arange(120, dtype=np.float64).reshape(2, 2, 5, 6)[:, :, ::-1, ::2]
    assert not data.flags.c_contiguous

    actual = F.im2col(data, (2, 3), (2, 1), (1, 0), to_matrix)

    np.testing.assert_array_equal(
        actual.data, reference_patches(data, (2, 3), (2, 1), (1, 0), to_matrix)
    )


@pytest.mark.parametrize("shape,kernel_size,stride,pad", CASES)
@pytest.mark.parametrize("to_matrix", [False, True])
def test_col2im_matches_patch_extraction_adjoint(shape, kernel_size, stride, pad, to_matrix):
    # Each basis image selects exactly the columns contributing to one pixel.
    template = reference_patches(np.zeros(shape), kernel_size, stride, pad, to_matrix)
    columns = np.linspace(-1, 2, template.size).reshape(template.shape)
    original = columns.copy()
    expected = np.zeros(shape)
    for index in np.ndindex(shape):
        basis = np.zeros(shape)
        basis[index] = 1
        expected[index] = np.sum(
            reference_patches(basis, kernel_size, stride, pad, to_matrix) * columns
        )

    actual = col2im_array(columns, shape, kernel_size, stride, pad, to_matrix)
    wrapped = F.col2im(columns, shape, kernel_size, stride, pad, to_matrix)

    assert isinstance(wrapped, Variable)
    assert actual.dtype == wrapped.dtype == columns.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(wrapped.data, expected, atol=1e-14)
    np.testing.assert_array_equal(columns, original)


@pytest.mark.parametrize("to_matrix", [False, True])
def test_col2im_sums_overlapping_patches(to_matrix):
    data = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    columns = F.im2col(data, 2, to_matrix=to_matrix)

    restored = F.col2im(columns, data.shape, 2, to_matrix=to_matrix)

    counts = np.array([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=np.float32)
    assert restored.dtype == data.dtype
    np.testing.assert_array_equal(restored.data, data * counts)


def numerical_gradient(function, data, eps=1e-5):
    gradient = np.zeros_like(data)
    for index in np.ndindex(data.shape):
        above, below = data.copy(), data.copy()
        above[index] += eps
        below[index] -= eps
        with no_grad():
            gradient[index] = (function(above).data - function(below).data) / (2 * eps)
    return gradient


@pytest.mark.parametrize("to_matrix", [False, True])
@pytest.mark.parametrize("operation", ["im2col", "col2im"])
def test_patch_operation_backward_matches_numerical_gradient(to_matrix, operation):
    shape = (1, 2, 3, 4)
    kernel_size, stride, pad = (2, 3), (2, 1), (1, 1)
    rng = np.random.default_rng(0)
    data = rng.normal(size=shape)
    if operation == "im2col":
        def transform(values):
            return F.im2col(values, kernel_size, stride, pad, to_matrix)
    else:
        data = reference_patches(data, kernel_size, stride, pad, to_matrix)

        def transform(values):
            return F.col2im(values, shape, kernel_size, stride, pad, to_matrix)

    weights = rng.normal(size=transform(data).shape)

    def objective(values):
        return (transform(values) * weights).sum()

    x = Variable(data.copy())
    objective(x).backward()

    assert x.grad.shape == data.shape
    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, data), atol=1e-8)


@pytest.mark.parametrize("to_matrix", [False, True])
def test_im2col_supports_second_derivatives(to_matrix):
    data = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
    x = Variable(data.copy())
    columns = F.im2col(x, 2, to_matrix=to_matrix)

    (columns ** 2).sum().backward(create_graph=True)

    counts = np.array([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]])
    first_gradient = x.grad
    np.testing.assert_array_equal(first_gradient.data, 2 * data * counts)
    x.cleargrad()
    first_gradient.sum().backward()
    np.testing.assert_array_equal(x.grad.data, 2 * counts)
