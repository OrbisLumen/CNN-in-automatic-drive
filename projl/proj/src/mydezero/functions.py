import numpy as np
from mydezero.core import Function, Variable


class Square(Function):
    """Computes the element-wise square of the input."""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * gy * x
        return gx


class Exp(Function):
    """Computes the element-wise exponential of the input."""

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def square(x): return Square()(x)


def exp(x): return Exp()(x)
