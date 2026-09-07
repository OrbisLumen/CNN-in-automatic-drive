import numpy as np
from .core import Function

class Square(Function):
    def forward(self, x):
        return x ** 2
