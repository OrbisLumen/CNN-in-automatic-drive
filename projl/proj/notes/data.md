# 数据集、变换与批次训练

## Dataset 与 MNIST

`Dataset(train=True, transform=None, target_transform=None)` 在初始化时调用子类 `prepare()`，由子类设置 `self.data` 和可选的 `self.label`。`len(dataset)` 返回样本数；`dataset[index]` 接受标量索引，按需应用两类变换并返回 `(sample, label)`。无标签时返回 `(sample, None)`；None 变换等价于不变换。

```python
import numpy as np
from mydezero import Dataset, DataLoader

class TinyDataset(Dataset):
    def prepare(self):
        self.data = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.label = np.array([0, 1, 0, 1, 0, 1])

loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
assert [len(t) for x, t in loader] == [4, 2]
```

`mydezero.datasets.MNIST(train=True)` 选择 60,000 张训练图片；`train=False` 选择 10,000 张测试图片。首次使用下载 Google 托管的 MNIST GZIP 文件至 `~/.mydezero`，再次使用复用缓存。HTTPS 需要 Python 的 CA 证书配置正常。

初始化时解压整个 split 至内存：

| 内容 | 原始存储 | `dataset[index]` 默认返回 |
| --- | --- | --- |
| 图像 | `uint8`，形状 `(N, 1, 28, 28)` | `float32`，形状 `(784,)`，范围 `[0, 1]` |
| 标签 | `uint8`，形状 `(N,)` | 0–9 的类别索引 |

训练和测试原始像素约占 44.9 MiB 与 7.5 MiB。解压时有临时内存开销。原始数组来自不可变 bytes 的 NumPy 视图，通常只读；默认变换在取样时创建浮点数据，不会把整个存储提前转为 float32。修改原始数据前需要自行复制。

```python
from mydezero.datasets import MNIST

train = MNIST(train=True)
x, label = train[0]
assert x.shape == (784,)
train.show(row=2, col=5)  # 随机抽取十次，允许重复，显示原始灰度图
assert MNIST.labels()[3] == '3'

raw = MNIST(train=False, transform=None)
assert raw[0][0].shape == (1, 28, 28)
```

`show` 不使用样本 transform，不显示标签；`labels()` 返回整数类别到字符名称的映射。

## 样本变换

从 `mydezero.transforms` 导入：

| 变换 | 用法 |
| --- | --- |
| `Compose(transforms=[])` | 按列表顺序执行；空组合原样返回输入 |
| `Flatten()` | 展平为一维数组 |
| `AsType(dtype=np.float32)` | 转换 NumPy dtype |
| `ToFloat(dtype=np.float32)` | AsType 的别名 |
| `ToInt(dtype=int)` | 默认转为平台整数，也可指定整数 dtype |
| `Normalize(mean=0, std=1)` | `(array - mean) / std`；std 应非零，序列参数沿第一个轴按通道广播 |

对于图像，先转浮点再归一化，避免整数运算造成截断或溢出。按通道归一化使用 `(C, H, W)`，应在 Flatten 之前执行。

```python
from mydezero.transforms import Compose, Flatten, Normalize, ToFloat

transform = Compose([ToFloat(), Normalize([0.5], [0.5]), Flatten()])
# 上述均值和标准差假定输入已处于 [0, 1]；MNIST 原始像素是 [0, 255]。
mnist_transform = Compose([Flatten(), ToFloat(), Normalize(0, 255)])
```

## DataLoader 与评估

`DataLoader(dataset, batch_size, shuffle=True)` 每次返回 `(x, t)` 两个 NumPy 数组，包括最后不足 batch_size 的批次。batch_size 必须为正整数。`shuffle=False` 保留顺序；默认开始及每次耗尽后重新生成随机排列。

迭代器耗尽时自动 `reset()`，因此可直接跨 epoch 复用。中途 break 不会重置，下次从未读批次继续；要重新开始需手动 `loader.reset()`。`loader.next()` 等价于 `next(loader)`。空数据集不产生批次。无标签数据会返回含 None 的 object 标签数组，不能直接用于监督损失。

```python
import mydezero
import mydezero.functions as F
from mydezero.models import MLP
from mydezero.optimizers import SGD

model = MLP((16, 2))
optimizer = SGD(lr=0.1).setup(model)
loader = DataLoader(TinyDataset(), batch_size=4, shuffle=True)
for epoch in range(2):
    for x, t in loader:
        loss = F.softmax_cross_entropy(model(x), t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

correct = total = 0
with mydezero.no_grad():
    for x, t in loader:
        correct += float(F.accuracy(model(x), t).data) * len(t)
        total += len(t)
print(correct / total)
```

汇总损失或正确率时按各批实际大小加权，并除以当前评估数据集的样本总数。测试集不能使用训练集数量作分母。完整 MNIST 示例见 [train_mnist_mlp.ipynb](../examples/neural_network/train_mnist_mlp.ipynb)。

## 下载缓存约定

`get_file` 展开 cache_dir 中的 `~` 并转为绝对路径，支持嵌套目录和指定文件名。默认从 URL 路径提取文件名，排除 query 和 fragment。调用方应提供带有效文件名的 URL，或显式指定 file_name。

缓存存在时不会重新请求，也不会校验来源、大小或内容。同名文件共用缓存路径；需区分时传入不同 file_name 或 cache_dir。下载失败或 Ctrl+C 会清理当前未完成文件并抛出原异常；不提供断点续传。`show_progress` 显示百分比或未知总大小时的字节数，最后不足一块时最多显示 100%。
