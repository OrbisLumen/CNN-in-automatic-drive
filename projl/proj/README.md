# MyDeZero Project

This project contains a small automatic differentiation library inspired by
DeZero. The source code lives in `src/mydezero`, and the test suite checks core
features such as variables, functions, backpropagation, and configuration.

## Project Structure

```text
projl/proj/
├── src/
│   └── mydezero/      # Library source code
├── tests/             # Pytest test suite
└── README.md
```

## Get Started
Run the following command first from the `projl/proj` directory.

```bash
pip3 install -e .
```

This make sure the jupyter and others know which package is mydezero.

## Run Tests

Run the tests from the `projl/proj` directory:

```bash
cd "CNN-in-automatic-drive/projl/proj"
python3 -m pytest tests
```

The test configuration automatically adds `src/` to Python's import path, so no
extra environment variables are required.

## Implementation Compared with DeZero

The local reference is `../reference/deep-learning-from-scratch-3-master-cn/dezero`.
MyDeZero currently implements a NumPy subset of that framework:

| Module | Current implementation | Scope compared with DeZero |
| --- | --- | --- |
| `core.py` | Variable, Parameter, Function, graph configuration, arithmetic, higher-order differentiation | Uses a generation heap for backward traversal; NumPy only |
| `functions.py` | Elementary functions, reshape, transpose, reductions, broadcasting, matrix multiplication, linear transform, squared error, sigmoid | Matrix multiplication and linear transforms support 2D batches; squared error follows DeZero's batch-size normalization |
| `layers.py` | Recursive parameter collection and Linear with optional bias and deferred initialization | Convolution, recurrent layers, and weight serialization remain unimplemented |
| `models.py` | Model base class and graph visualization | Sequential, MLP, and pretrained networks remain unimplemented |
| `optimizers.py` | Optimizer base class, hooks, and SGD | Other update rules and built-in hooks remain unimplemented |
| `utils.py` | DOT graph generation, rendering, and reduction gradient helpers | Visualization lives here; `graph.py` is currently empty |

Library docstrings use Google style (`Args`, `Returns`, `Yields`, `Attributes`, and
`Raises` where relevant). Function subclasses inherit the forward/backward contract
from `Function`; simple overrides document the operation at the class level.

Assign parameters and child layers directly to attributes so `Layer.params()` can
collect them. Gradients are `Variable` objects; call `model.cleargrads()` before each
new backward pass and attach the optimizer with `SGD(...).setup(model)` before
calling `update()`.

Graph rendering requires the Graphviz `dot` executable. Rendered files and shared
temporary DOT source are stored in `draft/`; notebook display also uses IPython.
