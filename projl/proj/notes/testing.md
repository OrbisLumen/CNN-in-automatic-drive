# Tests and Coverage

Run from the project directory:

```bash
python -m pytest tests -q
```

Use the notebook's Python environment. Tests require NumPy, Matplotlib, and pytest.
The rendering test requires system Graphviz and skips without `dot`.
IPython supports notebook image display. The full suite contains **199 tests**,
including parameterized cases.

## Feature Coverage

| Feature | Test files | Checks |
| --- | --- | --- |
| Variable, Parameter, core wrappers | `core/test_variable.py`, `test_layers.py`, `test_models.py` | Properties, types, wrapper identity, parameter registration, gradient clearing |
| Operators and broadcasting | `core/test_arithmetic.py` | Forward/backward operations, scalars, arrays, broadcast gradient reduction |
| Automatic differentiation | `core/test_backward.py` | Branches, variable reuse, gradient accumulation, intermediate gradients, custom multi-output functions, finite differences |
| Configuration contexts | `core/test_config.py` | Graph disabling, nesting, restoration after exceptions |
| Elementary functions and activations | `functions/test_elementary.py` | square/exp/sin/cos/tanh, both sigmoids, ReLU at zero, higher-order derivatives, stability |
| Optimization examples | `functions/test_optimization.py` | sphere/matyas/goldstein gradients |
| Tensor operations | `functions/test_tensor.py` | reshape, explicit transpose, sum, sum_to, broadcast_to, indexing, repeated indices, higher-order indexing gradients |
| Matrix operations and regression loss | `functions/test_tensor.py`, `test_layers.py` | matmul, linear with/without bias, input/weight/bias gradients, MSE batch normalization |
| Classification | `functions/test_classification.py` | Softmax axes, cross entropy numerical gradients, second derivatives, large logits, underflow, single class, accuracy, column labels, no-gradient behavior |
| Extrema and clipping | `functions/test_extrema.py` | Positive/negative axes, retained dimensions, ties, boundaries, higher-order gradients |
| Layers and models | `test_layers.py`, `test_models.py` | Deferred initialization, optional bias, recursive parameters, Sequential, MLP, hidden activations, plot calls |
| Optimizers | `test_optimizers.py` | SGD, MomentumSGD, hooks, independent velocities, missing gradients, zero momentum |
| Dataset and MNIST | `test_datasets.py` | prepare, unlabeled data, transforms, train/test files, IDX headers, shapes/dtypes, raw storage, sample preview |
| DataLoader | `test_dataloaders.py` | Order, shuffle, partial batches, repeated epochs, reset, next, empty/unlabeled data |
| Transforms | `test_transforms.py` | Compose order and empty lists, flattening, casting, normalization, channel broadcasting, input preservation |
| Downloads and progress | `test_downloads.py` | File downloads, nested/default/custom caches, names, URL queries, cache reuse, failure/interruption cleanup, unknown totals, progress limits |
| Numerical and graph utilities | `test_utils.py`, `integration/test_graph_rendering.py` | Stable logsumexp, reduction shapes, DOT generation, actual PNG rendering |
| Training pipeline | `integration/test_training.py` | Model/optimizer combinations, decreasing loss, Dataset → DataLoader → classification training → weighted evaluation |

Paths are relative to `tests/`. Tests use known results, analytic gradients, finite
differences, and small training tasks. They check correctness, not just line coverage,
and do not cover every input combination.

## Reproducibility and External Dependencies

MNIST tests generate two-image IDX/GZIP fixtures with real headers and pixels,
then replace the download entry point to test parsing without network access or
user-cache changes. Download tests use temporary `file://` sources and simulate
network failures and Ctrl+C. Shuffle tests inject fixed index orders to check
sample coverage and label alignment.

The full MNIST notebook needs network access or cached data. Its five-epoch training
run is outside the default suite. Automated training tests use small deterministic
data to check parameter updates and accuracy. Preview tests check image pixels and
shape while replacing Matplotlib display calls. Graph rendering separately verifies
a real PNG file. GPU paths require CuPy and a CUDA device; they are not covered
by the automated suite.

Run Python snippets in each document from top to bottom. MNIST examples in
`data.md` download and load real data; `show()` displays images. When adding APIs,
update [api.md](api.md), relevant usage notes, and this table, and add behavior tests.
