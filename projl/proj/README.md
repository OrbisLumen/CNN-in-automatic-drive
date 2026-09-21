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
