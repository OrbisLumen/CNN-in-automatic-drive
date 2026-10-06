# CNN-in-automatic-drive

## Overview

This repository contains coursework and semester projects on the application of
convolutional neural networks (CNNs) to automated driving. The work is organized
into three semester directories:

- `projl`: first-semester work
- `projll`: second-semester work
- `projlll`: third-semester work

## Repository Structure

Shared documentation is maintained in `doc/`. Each semester has its
own requirements, reference materials, project implementation, and assignments.

```text
CNN-in-automatic-drive/
├── AGENTS.md      # Repository working guidelines
├── README.md     # Project overview and setup
├── doc/          # Shared documentation and policy
├── projl/        # First-semester work
├── projll/       # Second-semester work
└── projlll/      # Third-semester work
```

Each semester folder generally follows this structure:

```text
proj*/
├── reference/  # Reference materials and examples
├── request/    # Assignment requirements and instructions
├── proj/       # Main semester project
└── work/       # Completed and in-progress work
    └── hw*/    # Homework assignments
```

## Environment Setup

The semester projects currently use Python. Use an isolated environment, such as
`venv` or Conda, to manage project dependencies.

For a `venv` environment on macOS or Linux, run the following commands from the
repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy jupyter matplotlib pytest
```

The shared Python tools are:

- `numpy`: numerical computation.
- `jupyter`: notebook-based experiments.
- `matplotlib`: plotting and visualization.
- `pytest`: automated testing.

Refer to the relevant semester project README for additional setup instructions.
For the first-semester MyDeZero project, install the local package from the
repository root:

```bash
python3 -m pip install -e ./projl/proj
```

## Testing

Run the first-semester MyDeZero test suite from its project directory:

```bash
cd projl/proj
python3 -m pytest tests
```

See [the MyDeZero README](projl/proj/README.md) for individual test groups and
optional Graphviz support.

## Collaboration

The team uses personal development branches named `dev/<github-username>` and
pull requests targeting `master`. OrbisLumen and Booyean are the designated PR
reviewers and merge maintainers.

The [Git and GitHub collaboration policy](doc/collaboration.md) defines branch
naming, contributor responsibilities, commit conventions, review procedures, and
the proposed configuration for restricting updates to `master`. The document
distinguishes team policy from GitHub settings that still require implementation.

## Working Guidelines

- Read the applicable `request/` requirements before starting an assignment.
- Use the corresponding `reference/` directory for supporting materials.
- Keep assignments in the appropriate `work/hw*/` directory and semester project
  implementations in `proj/`.
- Keep changes scoped to the relevant task and semester.
- Consult [AGENTS.md](AGENTS.md) and any applicable directory-specific guidelines.
