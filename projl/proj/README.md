# MyDeZero Project

This project contains a small automatic differentiation library inspired by
DeZero. The source code lives in `src/mydezero`, and the test suite checks core
features such as variables, functions, backpropagation, and configuration.

## Project Structure

```text
projl/proj/
├── src/
│   └── mydezero/      # Library source code
├── tests/
│   ├── core/          # Variables, arithmetic, backward, configuration
│   ├── functions/     # Elementary, tensor, and optimization functions
│   ├── integration/   # Training steps and Graphviz rendering
│   ├── test_layers.py
│   ├── test_models.py
│   ├── test_optimizers.py
│   ├── test_utils.py
│   ├── conftest.py    # Import setup and shared model fixture
│   └── helpers.py     # Numerical differentiation helper
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

Run a feature group or an individual module while working on it:

```bash
python3 -m pytest tests/core
python3 -m pytest tests/functions/test_tensor.py
python3 -m pytest tests/test_optimizers.py
python3 -m pytest tests/integration
```

The Graphviz rendering test uses a temporary directory and skips automatically
when `dot` is unavailable. Other tests do not require Graphviz. Shared model
fixtures return a fresh instance for every test.

The tensor tests cover indexing with slices, scalar indices, boolean masks,
repeated indices, empty selections, and second-order gradients. Model and
optimizer tests check layer order, recursive parameter registration, hidden-layer
activations, persistent momentum, independent parameter velocities, and hooks.
Integration tests train both Sequential and MLP models with MomentumSGD and check
that the loss decreases over multiple steps.

## Indexing, Composed Models, and Momentum

Use `get_item` for differentiable NumPy indexing:

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

x = Variable(np.array([2.0, 3.0, 5.0]))
y = F.get_item(x, [0, 0, 2])
y.sum().backward()
np.testing.assert_array_equal(x.grad.data, [2.0, 0.0, 1.0])
```

Repeated indices accumulate gradients at the selected positions; unselected
positions receive zero. Use `backward(create_graph=True)` when computing
higher-order derivatives. Call `F.get_item(x, indices)` explicitly; `Variable`
currently has no `x[indices]` operator.

`Sequential` applies its layers in order and registers them for parameter
collection and gradient clearing. An empty Sequential returns its input.
`MLP` accepts a nonempty sequence of output widths, infers input widths on its
first call, and applies sigmoid between hidden layers by default. Its last layer
has no activation. Pass `activation=F.tanh` to select another differentiable
hidden-layer activation.

```python
from mydezero.layers import Linear
from mydezero.models import MLP, Sequential
from mydezero.optimizers import MomentumSGD

# Two affine layers; Sequential does not insert activations.
sequence = Sequential(Linear(4), Linear(1))
assert sequence(np.zeros((2, 3), dtype=np.float32)).shape == (2, 1)

# One hidden layer with sigmoid and one affine output layer.
model = MLP((4, 1))
optimizer = MomentumSGD(lr=0.01, momentum=0.9).setup(model)
inputs = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
targets = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)

for _ in range(100):
    model.cleargrads()
    loss = F.mean_squared_error(model(inputs), targets)
    loss.backward()
    optimizer.update()
```

MomentumSGD stores one velocity per parameter, initialized with zeros on its
first update. It computes `v = momentum * v - lr * grad`, then `param += v`.
Reuse the optimizer across training steps to retain momentum; creating a new
optimizer resets its state. Parameters without gradients are skipped, retaining
their velocity. Hooks run before updates, as with SGD. A momentum of zero produces
the same updates as SGD. Trainable parameters should use floating-point NumPy
arrays.

## Implementation Compared with DeZero

The local reference is `../reference/deep-learning-from-scratch-3-master-cn/dezero`.
MyDeZero currently implements a NumPy subset of that framework:

| Module | Current implementation | Scope compared with DeZero |
| --- | --- | --- |
| `core.py` | Variable, Parameter, Function, graph configuration, arithmetic, higher-order differentiation | Uses a generation heap for backward traversal; NumPy only |
| `functions.py` | Elementary functions, reshape, transpose, indexing, reductions, broadcasting, matrix multiplication, linear transform, squared error, sigmoid | NumPy indexing supports repeated-index gradient accumulation and higher-order differentiation; matrix multiplication and linear transforms support 2D batches; squared error follows DeZero's batch-size normalization |
| `layers.py` | Recursive parameter collection and Linear with optional bias and deferred initialization | Convolution, recurrent layers, and weight serialization remain unimplemented |
| `models.py` | Model base class, graph visualization, Sequential, and MLP | Pretrained networks remain unimplemented |
| `optimizers.py` | Optimizer base class, hooks, SGD, and MomentumSGD | Other update rules and built-in hooks remain unimplemented |
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
