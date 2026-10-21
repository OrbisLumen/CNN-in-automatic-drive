# Jupyter Examples

From `projl/proj/`, install the library in the notebook kernel's Python environment:

```bash
python -m pip install -e .
```

Examples use NumPy. Plots require Matplotlib; computation graphs require IPython
and system Graphviz (`dot`). Select the matching Python 3 kernel and run cells in order.

Suggested reading order:

| Category | Notebook | Topics |
| --- | --- | --- |
| `autograd` | [draw_goldstein_graph.ipynb](autograd/draw_goldstein_graph.ipynb) | Function evaluation, backpropagation, and computation graphs |
| `autograd` | [draw_sin_higher_order_derivatives.ipynb](autograd/draw_sin_higher_order_derivatives.ipynb) | First through third derivatives of sine |
| `autograd` | [optimize_by_newtons_method.ipynb](autograd/optimize_by_newtons_method.ipynb) | Newton's method with second derivatives |
| `regression` | [regression_simple.ipynb](regression/regression_simple.ipynb) | Manual linear regression training |
| `regression` | [nonlinear_regression.ipynb](regression/nonlinear_regression.ipynb) | Fitting nonlinear data with a two-layer network |
| `neural_network` | [using_model_simple.ipynb](neural_network/using_model_simple.ipynb) | Defining a network with Model and Linear |
| `neural_network` | [train_mnist_mlp.ipynb](neural_network/train_mnist_mlp.ipynb) | Training an MLP on MNIST |

Computation graphs are saved in the project's `draft/` directory.
Notebooks retain their saved outputs; rerun them after code changes.
