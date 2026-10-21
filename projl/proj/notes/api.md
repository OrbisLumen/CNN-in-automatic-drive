# MyDeZero API and Feature Index

MyDeZero provides NumPy automatic differentiation and fully connected network training.
`Function` subclasses implement operations; normally call their lowercase wrappers.
See [usage](usage.md), [data](data.md), and [examples](../examples/README.md).

## Core and Automatic Differentiation

Import core objects from `mydezero`:

| API | Usage and conventions |
| --- | --- |
| `Variable(data, name=None)` | Wraps a NumPy array; `None` means uninitialized. Wrap Python scalars with `np.array` first |
| `Parameter(data, name=None)` | A Variable that Layer can register automatically |
| `data`, `grad`, `creator`, `generation` | Raw data, gradient, creating function, and graph depth; gradients are Variables |
| `shape`, `ndim`, `size`, `dtype`, `len(x)` | Underlying array properties |
| `backward(retain_grad=False, create_graph=False)` | Starts with an all-ones gradient by default; repeated calls accumulate leaf gradients. Enable flags to retain intermediate gradients or build a gradient graph |
| `cleargrad()` | Clears this variable's gradient; use `model.cleargrads()` for model parameters |
| `reshape(*shape)`, `transpose(*axes)`, `.T`, `sum(axis=None, keepdims=False)` | Differentiable shape operations and summation |
| `+`, `-`, `*`, `/`, unary `-`, `**` | NumPy broadcasting and scalars on either side; exponents are constants |
| `using_config(name, value)` | Temporarily sets a Config attribute; restores it after nesting or exceptions |
| `no_grad()` | Disables graph recording for inference or evaluation; leaves parameter values unchanged |
| `Config.enable_backprop` | Controls graph recording; prefer context managers over direct changes |
| `Function` | Implement `forward` and `backward`; calling an instance connects the graph |

`mydezero.core.as_array` converts scalars to arrays. `as_variable` wraps arrays and
preserves existing Variable identity. Package import calls `setup_variable()` to
register operators; manual calls are usually unnecessary.

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

## Functions

Import from `mydezero.functions`, usually as `F`. Inputs accept Variables or NumPy
arrays; outputs are Variables.

| API | Behavior and limits |
| --- | --- |
| `square(x)`, `exp(x)`, `sin(x)`, `cos(x)`, `tanh(x)` | Elementwise functions with higher-order derivatives |
| `sphere(x, y)`, `matyas(x, y)`, `goldstein(x, y)` | Two-input optimization examples |
| `reshape(x, shape)`, `transpose(x, axes=None)` | Shape operations; backward restores input shape and axes |
| `get_item(x, slices)` | NumPy indexing; repeated indices accumulate gradients. No `x[slices]` syntax yet |
| `sum(x, axis=None, keepdims=False)` | Sums over the specified axes |
| `sum_to(x, shape)` | Sums broadcast dimensions to the target shape |
| `broadcast_to(x, shape)` | Broadcasts to a shape; backward sums gradients to the original shape |
| `matmul(x, W)` | 2D matrix multiplication |
| `linear(x, W, b=None)` | 2D batch affine transform `x @ W + b`; owns no parameters |
| `mean_squared_error(x0, x1)` | Nonempty batches with matching shapes; divides total squared error by batch size N, not element count |
| `softmax_cross_entropy(x, t)` | `(N, C)` logits and `(N,)` or `(N, 1)` integer labels; mean batch loss. Labels have no gradient |
| `accuracy(y, t)` | Same input shapes as cross entropy; no gradient graph. Argmax ties select the first class |
| `relu(x)` | Elementwise max(x, 0); gradient is 0 for negative values and zero, 1 for positive values |
| `sigmoid(x)` | Stable tanh-based sigmoid |
| `sigmoid_simple(x)` | Teaching implementation using basic operations; extreme negative inputs may overflow |
| `softmax(x, axis=1)` | Normalizes along axis; use `axis=0` for 1D input |
| `max(x, axis=None, keepdims=False)`, `min(...)` | Each tied extremum receives the full upstream gradient |
| `clip(x, x_min, x_max)` | Gradient is 1 inside the interval and at its boundaries, 0 outside |

Pass logits directly to cross entropy, without applying softmax first.
Labels are class indices, not one-hot vectors. See [usage.md](usage.md) for details.

## Layers, Models, and Optimizers

| API | Behavior |
| --- | --- |
| `Layer` | Implement `forward`. Assign Parameters and child layers directly to attributes for registration; ordinary containers are not scanned |
| `params()`, `cleargrads()` | Recursively collect parameters and clear gradients; traversal order is unspecified |
| `layers.Linear(out_size, nobias=False, dtype=np.float32, in_size=None)` | Owns W and optional b; infers input width on the first call if in_size is omitted |
| `Model` | Extends Layer with graph plotting |
| `Model.plot(*inputs, to_file='model.png')` | Runs forward and plots; paths are relative to the project's `draft/` |
| `models.Sequential(*layers)` | Calls layers in order without adding activations; an empty sequence returns its input |
| `models.MLP(fc_output_sizes, activation=F.sigmoid)` | Nonempty width sequence ending with output width; activates hidden layers only |
| `optimizers.Optimizer` | Implement `update_one(param)`; `setup(model)` binds the model and returns the optimizer |
| `add_hook(callback)` | Runs before updates in registration order; receives parameters with gradients |
| `update()` | Updates parameters with gradients; does not clear gradients |
| `optimizers.SGD(lr=0.01)` | `param -= lr * grad` |
| `optimizers.MomentumSGD(lr=0.01, momentum=0.9)` | Stores per-parameter velocity; `v = momentum*v - lr*grad`, `param += v` |

Use floating-point parameter arrays. Reuse MomentumSGD across batches to retain
momentum. Convolution, GPU, Dropout, weight saving, and pretrained networks are not implemented.

## Data and Utilities

See [data.md](data.md) for Dataset, MNIST, DataLoader, and transforms.

| `mydezero.utils` API | Behavior |
| --- | --- |
| `get_file(url, file_name=None, cache_dir=None)` | Download cache, default `~/.mydezero`; returns an absolute path. Removes partial files and re-raises on failure or interruption |
| `show_progress(block_num, block_size, total_size)` | urlretrieve callback; shows downloaded bytes if total size is unknown or zero |
| `get_dot_graph(output, verbose=True)` | Returns DOT text; verbose includes shape and dtype |
| `plot_dot_graph(output, verbose=True, to_file='graph.png')` | Renders with system Graphviz; returns an Image for notebooks, or None without IPython |
| `sum_to(x, shape)` | NumPy shape reduction without an autograd graph |
| `reshape_sum_backward(gy, x_shape, axis, keepdims)` | Restores removed axes for sum backward broadcasting |
| `logsumexp(x, axis=1)` | Stable NumPy log-sum-exp; retains reduced dimensions and preserves input |
| `max_backward_shape(x, axis)` | Shape for max/min backward broadcasting |

Rendered images and shared `temp.dot` go to the project's `draft/`; repeated plots
can overwrite files. `graph.py` is empty and has no separate API.
See [testing.md](testing.md) for test commands and coverage.
