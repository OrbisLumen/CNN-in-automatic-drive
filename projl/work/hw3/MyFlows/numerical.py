import numpy as np
from .core import Variable


def numerical_diff(f, x, eps=1e-4):
    """只支持标量输入和标量输出，另建变量，不修改 x。"""
    raise NotImplementedError("任务 T1：中心差分")


def gradcheck(f, x, eps=1e-4, atol=1e-6, rtol=1e-4):
    # 每次检查独立建图，避免旧梯度污染；第三讲必须手动设置种子。
    probe = Variable(x.data.copy())
    y = f(probe)
    y.grad = np.ones_like(y.data)
    y.backward()
    num = numerical_diff(f, x, eps)
    error = float(abs(probe.grad - num))
    return error <= atol + rtol * abs(float(num)), error
