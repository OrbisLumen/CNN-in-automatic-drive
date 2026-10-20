# MyDeZero API 与功能索引

本项目实现 NumPy 上的自动微分与全连接网络训练。以下列出当前公开功能；`Function` 子类负责运算实现，通常使用对应的小写函数调用。完整示例见 [usage.md](usage.md)、[数据流水线](data.md) 和 [examples](../examples/README.md)。

## 核心与自动微分

从 `mydezero` 导入核心对象：

| API | 用法与约定 |
| --- | --- |
| `Variable(data, name=None)` | 包装 NumPy 数组；支持 `None` 表示尚未初始化。Python 标量先用 `np.array` 包装 |
| `Parameter(data, name=None)` | 可被 Layer 自动登记的 Variable |
| `data`, `grad`, `creator`, `generation` | 原始数据、梯度、创建该变量的函数和图深度；梯度本身是 Variable |
| `shape`, `ndim`, `size`, `dtype`, `len(x)` | 查询底层数组属性 |
| `backward(retain_grad=False, create_graph=False)` | 默认以全一梯度反向传播；重复反向会累积叶子梯度。保留中间梯度或建立梯度计算图需显式开启 |
| `cleargrad()` | 清除本变量梯度；模型所有参数使用 `model.cleargrads()` |
| `reshape(*shape)`, `transpose(*axes)`, `.T`, `sum(axis=None, keepdims=False)` | 支持梯度的形状变换与求和 |
| `+`, `-`, `*`, `/`, 一元 `-`, `**` | 支持 NumPy 广播和左右侧标量运算；幂的指数为常数 |
| `using_config(name, value)` | 临时切换 Config 属性；嵌套和异常退出时恢复原值 |
| `no_grad()` | 暂停记录计算图，适合推理和评估；不改变参数值 |
| `Config.enable_backprop` | 控制是否记录计算图，通常使用上下文管理器而非直接修改 |
| `Function` | 自定义运算：实现 `forward` 和 `backward`，通过实例调用连接计算图 |

`mydezero.core.as_array` 将标量转换成数组，`as_variable` 包装数组并保留已有 Variable 的身份。`setup_variable()` 在包导入时注册运算符，通常不需手动调用。

```python
import numpy as np
from mydezero import Variable, no_grad

x = Variable(np.array(2.0))
y = x ** 3
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()
gx.backward()
np.testing.assert_allclose(x.grad.data, 12.0)
with no_grad():
    prediction = x * 2
assert prediction.creator is None
```

## 运算函数

以下函数从 `mydezero.functions`（通常缩写为 `F`）调用，输入接受 Variable 或 NumPy 数组，返回 Variable。

| API | 行为与限制 |
| --- | --- |
| `square(x)`, `exp(x)`, `sin(x)`, `cos(x)`, `tanh(x)` | 逐元素函数，支持高阶求导 |
| `sphere(x, y)`, `matyas(x, y)`, `goldstein(x, y)` | 二元优化示例函数 |
| `reshape(x, shape)`, `transpose(x, axes=None)` | 形状变换，反向还原输入轴和形状 |
| `get_item(x, slices)` | NumPy 索引；重复索引累加梯度。当前没有 `x[slices]` 语法 |
| `sum(x, axis=None, keepdims=False)` | 沿指定轴求和 |
| `sum_to(x, shape)` | 将广播扩张的维度求和到目标形状 |
| `broadcast_to(x, shape)` | 广播到目标形状，反向将梯度求和回原形状 |
| `matmul(x, W)` | 二维矩阵乘法 |
| `linear(x, W, b=None)` | 二维批次仿射变换 `x @ W + b`；不拥有参数 |
| `mean_squared_error(x0, x1)` | 同形状的非空批次；平方差总和除以批次大小 N，不除以元素总数 |
| `softmax_cross_entropy(x, t)` | `(N, C)` logits 与 `(N,)` 或 `(N, 1)` 整数标签；返回批次平均损失。标签不求导 |
| `accuracy(y, t)` | 分类正确率，输入形状同交叉熵；无梯度图，argmax 并列时取第一个类别 |
| `relu(x)` | 逐元素取 max(x, 0)，负数和零点梯度为 0，正数梯度为 1 |
| `sigmoid(x)` | 基于 tanh 的稳定 Sigmoid 实现 |
| `sigmoid_simple(x)` | 用基础运算组合的教学实现；极端负输入可能指数溢出 |
| `softmax(x, axis=1)` | 沿 axis 归一化；一维输入传 `axis=0` |
| `max(x, axis=None, keepdims=False)`, `min(...)` | 每个并列极值都获得完整上游梯度 |
| `clip(x, x_min, x_max)` | 区间内及边界梯度为 1，区间外为 0 |

分类训练直接将 logits 传入交叉熵，不要先做 Softmax。标签为类别索引而非 one-hot。更多数值和梯度约定见 [usage.md](usage.md)。

## 层、模型与优化器

| API | 行为 |
| --- | --- |
| `Layer` | 子类实现 `forward`。将 Parameter 或子层直接赋给属性以注册；普通容器内对象不会自动注册 |
| `params()`, `cleargrads()` | 递归遍历参数与清除梯度；遍历顺序不固定 |
| `layers.Linear(out_size, nobias=False, dtype=np.float32, in_size=None)` | 持有 W 和可选 b；省略 in_size 时首次前向推断输入宽度 |
| `Model` | 继承 Layer，增加计算图绘制功能 |
| `Model.plot(*inputs, to_file='model.png')` | 前向计算并绘图；文件相对项目 `draft/` 目录 |
| `models.Sequential(*layers)` | 按顺序调用；不自动插入激活函数，空序列直接返回输入 |
| `models.MLP(fc_output_sizes, activation=F.sigmoid)` | 非空宽度序列，最后一项是输出宽度；只在隐藏层后激活 |
| `optimizers.Optimizer` | 子类实现 `update_one(param)`；`setup(model)` 绑定目标并返回自身 |
| `add_hook(callback)` | 按登记顺序在更新前调用，接收有梯度的参数列表 |
| `update()` | 更新有梯度的参数，不自动清除梯度 |
| `optimizers.SGD(lr=0.01)` | `param -= lr * grad` |
| `optimizers.MomentumSGD(lr=0.01, momentum=0.9)` | 持久维护逐参数速度；`v = momentum*v - lr*grad`，`param += v` |

训练参数使用浮点数组。MomentumSGD 必须跨批次复用才能保留动量。当前不包含卷积、GPU、Dropout、权重保存或预训练网络。

## 数据和工具

Dataset、MNIST、DataLoader 与所有 transforms 的参数、形状和示例见 [data.md](data.md)。

| `mydezero.utils` API | 行为 |
| --- | --- |
| `get_file(url, file_name=None, cache_dir=None)` | 本地下载缓存；默认 `~/.mydezero`，返回绝对路径；失败或中断时删除残留并重新抛异常 |
| `show_progress(block_num, block_size, total_size)` | urlretrieve 回调；总大小未知或为零时显示已下载字节数 |
| `get_dot_graph(output, verbose=True)` | 返回 DOT 文本；verbose 包含形状与 dtype |
| `plot_dot_graph(output, verbose=True, to_file='graph.png')` | 使用系统 Graphviz 渲染；返回可供 notebook 显示的 Image，IPython 不可用时返回 None |
| `sum_to(x, shape)` | NumPy 版本的形状归约，不创建自动微分图 |
| `reshape_sum_backward(gy, x_shape, axis, keepdims)` | 为 sum 的反向广播恢复被移除的轴 |
| `logsumexp(x, axis=1)` | 保留归约维度的稳定 NumPy log-sum-exp，不修改输入 |
| `max_backward_shape(x, axis)` | 计算 max/min 反向广播所需形状 |

渲染的图片和共享 `temp.dot` 写入项目 `draft/`，重复绘图会覆盖同名文件。`graph.py` 目前为空，没有独立 API。测试运行方式及功能对应关系见 [testing.md](testing.md)。
