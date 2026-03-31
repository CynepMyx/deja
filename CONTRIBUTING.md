# Contributing

## Setup

```bash
git clone https://github.com/CynepMyx/deja.git
cd deja
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

First test run downloads the embedding model (~117 MB).

## Tests

```bash
pytest -v
```

Tests use temporary directories and in-memory databases — no external services required.

## Code style

- Python 3.10+ (no walrus operator abuse, keep it readable)
- No docstrings on obvious functions
- `print(..., file=sys.stderr)` for logging — stdout is reserved for MCP JSON-RPC
- Follow existing patterns in the codebase

## Pull requests

1. Create a branch from `main`
2. Make your changes
3. Ensure all tests pass
4. Open a PR with a clear description of what and why
