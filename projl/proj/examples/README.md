# Jupyter 示例

先在项目目录 `projl/proj/` 中，将库安装到 notebook 内核使用的 Python 环境：

```bash
python -m pip install -e .
```

示例使用 NumPy；绘图需要 Matplotlib，计算图显示需要 IPython 和系统 Graphviz（`dot` 命令）。选择对应的 Python 3 内核，按单元格顺序运行。

建议阅读顺序：

| 分类 | Notebook | 内容 |
| --- | --- | --- |
| `autograd` | [draw_goldstein_graph.ipynb](autograd/draw_goldstein_graph.ipynb) | 函数求值、反向传播与计算图 |
| `autograd` | [draw_sin_higher_order_derivatives.ipynb](autograd/draw_sin_higher_order_derivatives.ipynb) | 正弦函数的一至三阶导数 |
| `autograd` | [optimize_by_newtons_method.ipynb](autograd/optimize_by_newtons_method.ipynb) | 使用二阶导数进行牛顿法优化 |
| `regression` | [regression_simple.ipynb](regression/regression_simple.ipynb) | 手动实现线性回归训练 |
| `regression` | [nonlinear_regression.ipynb](regression/nonlinear_regression.ipynb) | 使用两层网络拟合非线性数据 |
| `neural_network` | [using_model_simple.ipynb](neural_network/using_model_simple.ipynb) | 使用 Model 和 Linear 定义网络 |
| `train_mnist_mlp` | [train_mnist_mlp.ipynb](neural_network/train_mnist_mlp.ipynb) | 使用 mlp 训练拟合 mnist |


计算图图片写入项目的 `draft/` 目录。Notebook 保留了原有输出；修改代码后，请重新运行以更新结果。