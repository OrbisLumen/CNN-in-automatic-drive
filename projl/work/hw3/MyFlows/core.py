import numpy as np


class Variable:
    def __init__(self, data, name=None):
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim != 0:
            raise ValueError("本教学阶段仅支持标量，张量与广播在后续课程实现")
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        """第三讲：手动设置根梯度；仅验收链和无共享变量的树。"""
        if self.grad is None:
            raise ValueError("第三讲须先设置 y.grad = np.ones_like(y.data)")
        f = self.creator
        if f is None:
            return
        raise NotImplementedError("任务 T2：计算输入梯度并递归传播")



class Function:
    def __call__(self, *inputs):
        self.inputs = tuple(x if isinstance(x, Variable) else Variable(x) for x in inputs)
        self.generation = max(x.generation for x in self.inputs)
        self.output = Variable(self.forward(*(x.data for x in self.inputs)))
        self.output.set_creator(self)
        return self.output

    @property
    def input(self):
        if len(self.inputs) != 1:
            raise ValueError("多输入运算请使用 inputs")
        return self.inputs[0]

    def forward(self, *xs):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError
