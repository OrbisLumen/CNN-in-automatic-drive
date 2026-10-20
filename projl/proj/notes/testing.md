# 测试与覆盖范围

从项目目录运行：

```bash
python -m pytest tests -q
```

使用与 notebook 一致的 Python 环境。测试需要 NumPy、Matplotlib 和 pytest；实际计算图渲染测试需要系统 Graphviz，缺少 `dot` 时该项跳过，IPython 用于 notebook 图片显示。当前完整运行包含 **199 项测试**（包括参数化用例）。

## 功能与测试对应

| 功能 | 测试文件 | 验证内容 |
| --- | --- | --- |
| Variable、Parameter 与核心包装 | `core/test_variable.py`、`test_layers.py`、`test_models.py` | 属性、类型、包装身份、参数登记、梯度清除 |
| 运算符与广播 | `core/test_arithmetic.py` | 正反向运算、标量和数组、广播后梯度归约 |
| 自动微分 | `core/test_backward.py` | 分支与变量复用、梯度累积、保留中间梯度、自定义多输出函数、数值差分 |
| 配置上下文 | `core/test_config.py` | 禁用计算图、嵌套与异常退出恢复 |
| 基础函数与激活 | `functions/test_elementary.py` | square/exp/sin/cos/tanh、两种 sigmoid、ReLU 零点与高阶梯度、高阶导数、稳定性 |
| 优化示例函数 | `functions/test_optimization.py` | sphere/matyas/goldstein 梯度 |
| 张量操作 | `functions/test_tensor.py` | reshape、显式 transpose、sum、sum_to、broadcast_to、索引、重复索引、高阶索引梯度 |
| 矩阵与回归损失 | `functions/test_tensor.py`、`test_layers.py` | matmul、带或不带偏置的 linear、输入/权重/偏置梯度、MSE 批次归一化 |
| 分类函数 | `functions/test_classification.py` | softmax 轴、交叉熵数值梯度、二阶导数、大 logits、下溢、单类、accuracy、列标签与无梯度行为 |
| 极值与裁剪 | `functions/test_extrema.py` | 正负轴、保留维度、并列极值、边界与高阶梯度 |
| 层与模型 | `test_layers.py`、`test_models.py` | 延迟初始化、可选偏置、递归参数、Sequential、MLP、隐藏层激活、plot 调用 |
| 优化器 | `test_optimizers.py` | SGD、MomentumSGD、hook、独立速度状态、缺失梯度、零动量 |
| Dataset 与 MNIST | `test_datasets.py` | prepare、无标签、样本变换、训练/测试文件选择、IDX 头、形状/dtype、原始存储、单样本预览 |
| DataLoader | `test_dataloaders.py` | 顺序、打乱、尾批次、重复 epoch、reset、next、空与无标签数据 |
| 数据变换 | `test_transforms.py` | Compose 顺序与空组合、展平/转型/归一化、按通道广播、输入不被修改 |
| 下载与进度 | `test_downloads.py` | 本地下载、嵌套/默认/自定义缓存、文件名、URL 查询参数、缓存复用、失败/中断清理、未知总大小与进度上限 |
| 数值与计算图工具 | `test_utils.py`、`integration/test_graph_rendering.py` | 稳定 logsumexp、归约形状、DOT 生成、真实 PNG 渲染 |
| 完整训练路径 | `integration/test_training.py` | 模型与优化器组合、损失下降，以及 Dataset → DataLoader → 分类训练 → 加权评估 |

表内路径相对于 `tests/`。测试通过解析已知结果、解析梯度、有限差分和小型训练任务检查行为，不以行覆盖率代替正确性判断，也不宣称穷尽所有输入组合。

## 可重复与外部依赖

MNIST 测试现场生成两张图片的 IDX/GZIP 文件，包含真实文件头和像素数据，通过替换下载入口验证解析；不联网、不占用真实用户缓存。下载工具用 `file://` 临时文件验证实际下载，并模拟网络失败和 Ctrl+C。打乱测试注入确定的索引顺序，检查每轮样本覆盖和样本/标签对应关系。

完整 MNIST notebook 需要联网或已缓存数据，五轮真实训练不属于默认测试。自动化训练测试使用小型确定性数据验证参数更新和准确率。图像预览测试检查拼图像素和形状，替换 Matplotlib 的显示调用以避免弹窗；计算图渲染则独立验证真实 PNG 文件。

文档中的 Python 代码可按同一文件从上到下运行。`data.md` 中 MNIST 示例会下载和加载真实数据，`show()` 会显示图片。新增 API 时请同步 [api.md](api.md)、相关用法及上表，并添加能检验具体行为的测试。
