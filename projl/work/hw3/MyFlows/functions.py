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


class Add(Function):
    def forward(self, a, b):
        return a + b
    def backward(self, gy):
        raise NotImplementedError("任务 T3：加法反向")


class Mul(Function):
    def forward(self, a, b):
        return a * b
    def backward(self, gy):
        raise NotImplementedError("任务 T3：乘法反向")


# 每次调用创建独立运算对象，不能复用同一 Function 实例建不同节点。
def square(x): return Square()(x)
def exp(x): return Exp()(x)
def add(a, b): return Add()(a, b)
def mul(a, b): return Mul()(a, b)
