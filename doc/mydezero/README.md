# MyDeZero

MyDeZero is a small automatic differentiation library inspired by DeZero. Its
[source](../../src/mydezero/) and [tests](../../tests/) share the repository's
Python environment. See the [root README](../../README.md) for installation.

- [API and feature index](api.md)
- [Usage and gradient conventions](usage.md)
- [Datasets, transforms, and mini-batches](data.md)
- [Test coverage and verification](testing.md)
- [Jupyter examples](../../examples/README.md)
- [UML design](../uml/README.md)

Run `python -m pytest tests` from the repository root. To check a single feature,
pass its test directory or module, for example `python -m pytest tests/core`.
