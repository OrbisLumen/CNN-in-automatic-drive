import numpy as np


class Variable:
    def __init__(self, data, name=None):
        self.data = np.asarray(data)
        self.name = name
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            raise ValueError("V1 请先设置输出梯度，例如 y.grad = np.ones_like(y.data)")
        func = self.creator
        if func is not None:
            x = func.input
            x.grad = func.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input):
        if hasattr(self, "input"):
            raise ValueError("每次运算请创建新实例，避免覆盖旧图，例如 Square()(x)")
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError("子类需要定义 forward")

    def backward(self, gy):
        raise NotImplementedError("子类需要定义 backward")
