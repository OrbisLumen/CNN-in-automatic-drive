"""Differentiable NumPy operations and optimization benchmark functions."""

import numpy as np
from mydezero.core import Function, as_variable
from mydezero import utils


# =============================================================================
# Basic Two Functions: square, exp
# =============================================================================

class Square(Function):
    """Compute the elementwise square with a differentiable backward pass."""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * gy * x
        return gx


def square(x):
    """Compute the elementwise square.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Square()(x)


class Exp(Function):
    """Compute the elementwise exponential with a differentiable backward pass.

    Uses differentiable operations in backward to preserve higher-order gradients.
    """

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


def exp(x):
    """Compute the elementwise exponential.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Exp()(x)


# =============================================================================
# Test functions for optimization
# =============================================================================

def sphere(x, y):
    """Evaluate the sphere benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    return x ** 2 + y ** 2


def matyas(x, y):
    """Evaluate the Matyas benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    """Evaluate the Goldstein-Price benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


# =============================================================================
# Basic functions: sin, cos, tanh
# =============================================================================

class Sin(Function):
    """Compute the elementwise sine with a differentiable backward pass."""

    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    """Compute the elementwise sine.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Sin()(x)


class Cos(Function):
    """Compute the elementwise cosine with a differentiable backward pass."""

    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    """Compute the elementwise cosine.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Cos()(x)


class Tanh(Function):
    """Compute the elementwise hyperbolic tangent with a differentiable backward pass."""

    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    """Compute the elementwise hyperbolic tangent.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Tanh()(x)


# =============================================================================
# Tensor Operations: reshape, transpose
# =============================================================================
class Reshape(Function):
    """Change an input shape without changing its elements.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    """Reshape an input while retaining its gradient connection.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape with at most one inferred dimension (-1).

    Returns:
        Variable: Reshaped input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """Permute axes and apply the inverse permutation during backward.

    Args:
        axes (tuple[int, ...] or list[int] or None): Axis permutation. None reverses
            all axes. Negative axes are supported.

    Attributes:
        axes (tuple[int, ...] or list[int] or None): Requested axis order.
    """

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axs = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axs)


def transpose(x, axes=None):
    """Permute input axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axes (tuple[int, ...] or list[int] or None): Axis permutation; None reverses
            all axes.

    Returns:
        Variable: Input with its axes reordered.
    """
    return Transpose(axes)(x)


# =============================================================================
# sum, sum_to, broadcast_to, matmul, linear
# =============================================================================

class Sum(Function):
    """Reduce an input and broadcast its gradient back to the input shape.

    Args:
        axis (int or tuple[int, ...] or None): Axes to reduce; None reduces all.
        keepdims (bool): Retain reduced axes as dimensions of size one.

    Attributes:
        axis (int or tuple[int, ...] or None): Reduction axes.
        keepdims (bool): Whether reduced dimensions are retained.
        x_shape (tuple[int, ...]): Input shape saved during forward.
    """

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    """Sum elements along the requested axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axis (int or tuple[int, ...] or None): Axes to reduce; None reduces all.
        keepdims (bool): Retain reduced axes as dimensions of size one.

    Returns:
        Variable: Reduced sum.
    """
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    """Reduce broadcast dimensions to a target shape.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    """Sum elements over broadcast dimensions to obtain a target shape.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape.

    Returns:
        Variable: Reduced input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    """Broadcast an input to a target shape.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    """Broadcast an input to a compatible target shape.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape.

    Returns:
        Variable: Broadcast input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    """Multiply two matrices and propagate gradients to both operands."""

    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    """Multiply two two-dimensional matrices.

    Args:
        x (Variable or np.ndarray): Left matrix of shape (N, K).
        W (Variable or np.ndarray): Right matrix of shape (K, M).

    Returns:
        Variable: Matrix product of shape (N, M).
    """
    return MatMul()(x, W)


class Linear(Function):
    """Compute an affine transform and gradients for the input, weights, and bias.

    The bias is optional. Its backward gradient is reduced over the batch dimension.
    This Function performs the computation; layers.Linear owns trainable parameters.
    """

    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    """Apply an affine transform to a batch of inputs.

    Args:
        x (Variable or np.ndarray): Input matrix of shape (N, in_size).
        W (Variable or np.ndarray): Weight matrix of shape (in_size, out_size).
        b (Variable or np.ndarray or None): Optional bias of shape (out_size,).

    Returns:
        Variable: Transformed batch of shape (N, out_size).
    """
    return Linear()(x, W, b)


# =============================================================================
# loss_functions: mean_squared_error
# =============================================================================

class MeanSquaredError(Function):
    """Sum squared errors and normalize by the batch size, following DeZero.

    Inputs should have the same shape and at least one dimension. For multicolumn
    outputs, the loss sums over output features rather than averaging over them.
    """

    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    """Compute squared error summed over elements and divided by batch size.

    Args:
        x0 (Variable or np.ndarray): Predictions of shape (N, ...) with N > 0.
        x1 (Variable or np.ndarray): Targets with the same shape as x0.

    Returns:
        Variable: Scalar loss normalized by N, not by the total element count.
    """
    return MeanSquaredError()(x0, x1)


# =============================================================================
# activation function: sigmoid
# =============================================================================

def sigmoid_simple(x):
    """Compute the logistic sigmoid by composing differentiable operations.

    This teaching version constructs the graph for 1 / (1 + exp(-x)); use sigmoid
    for the formulation that avoids exponential overflow on large negative inputs.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise sigmoid with the same shape as x.
    """
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    """Compute the elementwise logistic sigmoid with a differentiable backward pass.

    Uses the tanh formulation from DeZero to avoid exponential overflow.
    """

    def forward(self, x):
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    """Compute the elementwise logistic sigmoid.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Sigmoid()(x)
