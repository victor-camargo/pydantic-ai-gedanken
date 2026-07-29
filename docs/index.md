# Project Gedanken

## Installation

```bash
pip install pydantic-ai-gedanken
```

## Usage

```bash
pydantic-ai-gedanken hello --name world
```

## Development

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
make install    # install package + dev deps and set up pre-commit hooks
make format     # auto-format and fix with ruff
make lint       # check formatting and lint
make typecheck  # run pyright
make test       # run pytest
make docs       # build the documentation
make            # format, lint, typecheck, testcov
```
