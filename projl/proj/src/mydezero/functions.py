import numpy as np
from mydezero.core import Function, as_variable


# =============================================================================
# Basic Two Functions: square, exp
# =============================================================================

class Square(Function):
    """Computes the element-wise square of the input."""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * gy * x
        return gx


class Exp(Function):
    """Computes the element-wise exponential of the input."""

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x.data) * gy
        return gx


def square(x): return Square()(x)


def exp(x): return Exp()(x)


# =============================================================================
# Test functions for optimization
# =============================================================================

def sphere(x, y):
    return x ** 2 + y ** 2


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


# =============================================================================
# Basic functions: sin, cos, tanh
# =============================================================================

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x): return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x): return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x): return Tanh()(x)


# =============================================================================
# Tensor Operations: reshape, transpose
# =============================================================================
class Reshape(Function):
    """Reshapes the input tensor.

    Attributes:
        shape (tuple): Shape of the target.
        x.shape (tuple): Shape of the input.
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
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """Transposes the input tensor.

    Attributes:
        axes (None | list | tuple): Axes to transpose.
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
    return Transpose(axes)(x)
