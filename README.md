# deja

[![CI](https://github.com/CynepMyx/deja/actions/workflows/ci.yml/badge.svg)](https://github.com/CynepMyx/deja/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0id2hpdGUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjYiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS41Ii8+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjIiLz48L3N2Zz4=)](https://modelcontextprotocol.io/)

> Semantic search over your Claude Code session history. Ask questions about past conversations by meaning, not just keywords.

**deja** is an [MCP server](https://modelcontextprotocol.io/) that indexes Claude Code JSONL sessions and provides hybrid search (vector + full-text) directly from Claude Code.

## How it works

```
~/.claude/projects/*/        deja index         index.db
     *.jsonl           ──────────────►    (SQLite + vec + FTS5)
                        embeddings
                                           │
                                           │  deja serve (MCP stdio)
                                           ▼
                                      Claude Code
                                    "search past sessions"
```

1. **Index** — parses JSONL session files, extracts conversation turns, embeds with `multilingual-e5-small`, stores in SQLite
2. **Serve** — MCP server opens the index and answers search queries via stdio transport

Search combines vector KNN (semantic similarity) and FTS5 (keyword matching) via Reciprocal Rank Fusion.

## Install

```bash
git clone https://github.com/CynepMyx/deja.git
cd deja
python -m venv .venv
.venv/Scripts/pip install -e .        # Windows
# .venv/bin/pip install -e .          # Linux/macOS
```

First run downloads the embedding model (~117 MB).

## Usage

### Build the index

```bash
deja index              # incremental — only new/changed files
deja index --reindex    # full rebuild
```

Scans all `~/.claude/projects/*/*.jsonl` files.

### Add to Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
"deja": {
    "type": "stdio",
    "command": "/path/to/deja/.venv/Scripts/deja.exe",
    "args": ["serve"],
    "env": {
        "PYTHONUNBUFFERED": "1"
    }
}
```

Restart Claude Code — deja will appear as a connected MCP server.

### MCP Tools

| Tool | Description |
|------|-------------|
| `search` | Hybrid semantic + keyword search across all sessions |
| `get_session` | Retrieve full context of a specific session by ID |

**search** parameters:
- `query` (string) — what to search for
- `limit` (int, default 10) — max results
- `project` (string, optional) — filter by project
- `date_from` / `date_to` (string, optional) — ISO date range

### Auto-indexing (optional)

Index automatically when a Claude Code session ends. Add a Stop hook to `~/.claude/settings.json`:

```json
"hooks": {
    "Stop": [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": "/path/to/deja/.venv/bin/deja index"
                }
            ]
        }
    ]
}
```

On Windows with Git Bash, wrap in a shell script:

```bash
#!/bin/bash
DEJA="/path/to/deja/.venv/Scripts/deja.exe"
[ -f "$DEJA" ] && "$DEJA" index >/dev/null 2>&1 &
```

```json
"command": "bash /path/to/deja-index.sh"
```

PID lock prevents concurrent indexers — safe with multiple sessions.

## Stack

- **[fastembed](https://github.com/qdrant/fastembed)** — ONNX embeddings (`intfloat/multilingual-e5-small`, 384-dim)
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** — vector KNN search in SQLite
- **SQLite FTS5** — full-text keyword search
- **[FastMCP](https://github.com/jlowin/fastmcp)** — MCP server framework

## Performance

| Metric | Value |
|--------|-------|
| Incremental index | < 30 sec |
| Search latency (warm) | < 500 ms |
| First search (cold start) | < 5 sec |
| RAM (search) | ~150 MB |
| RAM (indexing) | ~300 MB |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
