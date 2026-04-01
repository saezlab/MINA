# Contributing guide

This document summarizes the main workflows for contributing to MINA.

## Installing development dependencies

You need the runtime dependencies of the package plus the extra tooling for tests and documentation.

### Using uv

```bash
uv sync --all-extras
```

This creates or updates the local `.venv` and installs the development, test, and docs dependencies.

### Using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test,doc]"
```

## Code style

This project uses `ruff` and `pre-commit` to keep formatting and linting consistent.

Enable local hooks with:

```bash
pre-commit install
```

Then run them on demand with:

```bash
pre-commit run --all-files
```

## Writing tests

This package uses `pytest` for automated testing.

Run the full suite with `uv`:

```bash
uv run pytest
```

Or with an activated virtual environment:

```bash
pytest
```

## Continuous integration

GitHub Actions runs the test suite on pull requests.
The supported Python versions and test matrix are defined in `pyproject.toml`.

## Publishing a release

Before making a release, update the version number in `pyproject.toml` and follow Semantic Versioning:

1. Increment MAJOR for incompatible API changes.
2. Increment MINOR for backwards-compatible functionality.
3. Increment PATCH for backwards-compatible bug fixes.

After that, create a GitHub release with a `vX.Y.Z` tag.

## Writing documentation

The documentation stack uses MkDocs with Material for MkDocs, `mkdocstrings` for the Python API reference,
and `mkdocs-jupyter` for the notebooks in `docs/notebooks`.

When you add or change user-facing functionality, update the relevant documentation page and, where needed,
refresh the notebooks so their saved output matches the current code.

## Building the docs locally

### Using uv

```bash
uv run mkdocs build --clean --strict
uv run python -m webbrowser -t site/index.html
```

### Using pip

```bash
source .venv/bin/activate
mkdocs build --clean --strict
python -m webbrowser -t site/index.html
```
