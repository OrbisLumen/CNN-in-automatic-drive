import numpy as np
from .core import Function


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return 2 * self.input.data * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.input.data) * gy
