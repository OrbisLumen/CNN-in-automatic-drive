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
│   ├── functions/     # Elementary, tensor, classification, extrema, benchmarks
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
python3 -m pytest tests/functions/test_classification.py
python3 -m pytest tests/functions/test_extrema.py
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

Classification tests compare softmax and cross entropy gradients with centered
finite differences, check second-order derivatives, and cover float32/float64,
column labels and large logits. Extrema tests cover positive,
negative, and tuple axes, retained dimensions, and tied values. Clipping tests
check the gradient convention at interval boundaries. A classification training
test verifies that cross entropy works with MLP and MomentumSGD.
