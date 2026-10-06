# Usage and Implementation Notes

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
import numpy as np
import mydezero.functions as F
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

## Softmax and Classification Loss

`F.softmax(x, axis=1)` converts an `(N, C)` batch of logits into class
probabilities. It subtracts the maximum before exponentiation to avoid overflow.
For a 1D input, specify `axis=0`. Other NumPy axes, including negative axes,
tuples, and `None`, are also supported.

Pass unnormalized logits directly to `F.softmax_cross_entropy_loss(x, t)`.
The loss centers logits before computing log probabilities and returns the mean
negative log probability over the batch. Logits must have nonempty shape `(N, C)`;
labels must be integer class indices in `[0, C)` with shape `(N,)` or `(N, 1)`.
One-hot labels are not accepted. Backward computes `(softmax(x) - one_hot(t)) / N`
for logits; labels receive no gradient. Floating-point logits should be finite.

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

logits = Variable(np.zeros((2, 3), dtype=np.float32))
labels = np.array([[0], [2]], dtype=np.int64)
loss = F.softmax_cross_entropy_loss(logits, labels)
loss.backward()

np.testing.assert_allclose(loss.data, np.log(3), rtol=1e-6)
assert logits.grad.shape == (2, 3)
assert logits.grad.dtype == np.float32
probabilities = F.softmax(logits)
np.testing.assert_allclose(probabilities.data.sum(axis=1), 1)
```

Use the same `model.cleargrads()`, `loss.backward()`, and `optimizer.update()`
training steps as for squared error, replacing the loss with cross entropy and
setting the final model width to the number of classes. Call
`backward(create_graph=True)` to retain a differentiable gradient graph for
second-order derivatives.

## Extrema, Clipping, and Numerical Helpers

`F.max` and `F.min` accept `axis` and `keepdims` with NumPy semantics. Negative
axes and tuples are supported. Their backward convention sends the full incoming
gradient to every tied extremum rather than dividing it among ties.

`F.clip(x, x_min, x_max)` clamps values using NumPy without modifying its input.
The gradient passes through values inside the interval and exactly on either
boundary, and is zero outside it.

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

x = Variable(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
np.testing.assert_array_equal(F.max(x, axis=-1).data, [2.0])
np.testing.assert_array_equal(F.min(x, axis=-1).data, [-2.0])
clipped = F.clip(x, -1.0, 1.0)
clipped.sum().backward()
np.testing.assert_array_equal(clipped.data, [[-1, -1, 0, 1, 1]])
np.testing.assert_array_equal(x.grad.data, [[0, 1, 1, 1, 0]])
```

`utils.logsumexp(x, axis=1)` computes log-sum-exp on a NumPy array while retaining
reduced dimensions. It avoids exponential overflow and preserves the input, but
does not record an automatic differentiation graph. `utils.max_backward_shape`
restores reduced axes as size one for broadcasting max/min gradients.

## Implementation Compared with DeZero

The local reference is `../../reference/deep-learning-from-scratch-3-master-cn/dezero`.
MyDeZero currently implements a NumPy subset of that framework:

| Module | Current implementation | Scope compared with DeZero |
| --- | --- | --- |
| `core.py` | Variable, Parameter, Function, graph configuration, arithmetic, higher-order differentiation | Uses a generation heap for backward traversal; NumPy only |
| `functions.py` | Elementary functions, tensor operations, squared error, softmax cross entropy, sigmoid, softmax, max/min, and clip | NumPy indexing supports repeated-index gradient accumulation and higher-order differentiation; matrix multiplication and linear transforms support 2D batches; classification loss accepts integer labels and averages over the batch |
| `layers.py` | Recursive parameter collection and Linear with optional bias and deferred initialization | Convolution, recurrent layers, and weight serialization remain unimplemented |
| `models.py` | Model base class, graph visualization, Sequential, and MLP | Pretrained networks remain unimplemented |
| `optimizers.py` | Optimizer base class, hooks, SGD, and MomentumSGD | Other update rules and built-in hooks remain unimplemented |
| `utils.py` | DOT graph generation, rendering, reduction gradient helpers, and stable logsumexp | Numerical helpers use NumPy; visualization lives here; `graph.py` is currently empty |

Library docstrings use Google style (`Args`, `Returns`, `Yields`, `Attributes`, and
`Raises` where relevant). Function subclasses inherit the forward/backward contract
from `Function`; simple overrides document the operation at the class level.

Assign parameters and child layers directly to attributes so `Layer.params()` can
collect them. Gradients are `Variable` objects; call `model.cleargrads()` before each
new backward pass and attach the optimizer with `SGD(...).setup(model)` before
calling `update()`.

Graph rendering requires the Graphviz `dot` executable. Rendered files and shared
temporary DOT source are stored in `draft/`; notebook display also uses IPython.
