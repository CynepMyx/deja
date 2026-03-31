# deja — Semantic Search for Claude Code Sessions

## Overview

MCP server providing semantic search over Claude Code JSONL session history.
Allows Claude Code to search its own past sessions by meaning, not just keywords.

## Architecture

Two modes:
- **CLI** (`deja index`) — parses JSONL, builds embeddings, writes SQLite index
- **MCP server** (`deja serve`) — opens index, answers search queries

Index and search are separate processes. Index runs on demand or via cron.
MCP server starts via stdio transport. Embedding model loads in lifespan (once at startup).

## Stack

- Python 3.10+
- fastembed >= 0.6.0 (pinned) — ONNX embeddings, `intfloat/multilingual-e5-small` via `add_custom_model()`
- sqlite-vec (pinned version) — vector KNN search in SQLite
- SQLite FTS5 — keyword search (parallel, not fallback)
- FastMCP — MCP server framework, stdio transport

### fastembed details

`intfloat/multilingual-e5-small` is NOT in fastembed's built-in registry. Must register manually:

```python
from fastembed import TextEmbedding
from fastembed.text.text_embedding import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),
    dim=384,
    model_file="onnx/model.onnx"
)
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
```

- Model size: ~117MB ONNX (first run downloads from HuggingFace)
- Speed: ~200-500 sentences/sec on CPU
- `embed()` returns generator — always wrap in `list()`
- Windows: `if __name__ == "__main__"` guard required for parallel workers

### sqlite-vec details

- Pre-v1 (v0.1.8) — pin exact version in pyproject.toml
- KNN uses `WHERE embedding MATCH ? AND k = ?` — NOT `LIMIT` (broken in SQLite < 3.41)
- Vector UPDATE impossible — use DELETE + INSERT
- KNN + JOIN: use CTE (first KNN, then JOIN metadata)
- k limit: max 4096 per query
- Vectors serialized via `struct.pack("%sf" % len(v), *v)`

## Data Source

- `~/.claude/projects/*/` — all `*.jsonl` files
- Each line: JSON with type (user/assistant/summary), message.content, timestamp, uuid

## Parser

- Extracts user/assistant message pairs
- From message.content: text blocks as-is, tool_use as `[Tool: name] + key params (file_path, command)`
- tool_result: stored as raw text (up to 2000 chars) for FTS search, but NOT included in embeddings
- Embedding input = user text + assistant text + tool_use descriptions only (no tool_result noise)
- Skips: thinking blocks, base64 data, screenshots, system messages
- Metadata per message: session_id (from filename), timestamp, project_path, message_index
- Schema validation with graceful degradation: skip malformed lines, log warnings, never crash

## Chunking

- One chunk = one conversation turn (user question + assistant response)
- If chunk > 1500 chars — split with 200 char overlap at sentence boundaries (`. `, `\n\n`)
- Stored: chunk_text, embedding (384-dim float32), session_id, timestamp, project_path, message_index
- Deduplication by unique key `(session_id, message_index, split_index)` — same text in different sessions is NOT a duplicate (different context)

## Indexing

- `deja index` — CLI command
- First run: full scan of all JSONL files
- Subsequent runs: incremental by file offset tracking
- Metadata table `indexed_files`: path, last_offset, last_mtime, last_size
- On changed file: read only from last_offset, parse new lines, embed, insert
- Safe reindex: if file size < last_offset (truncation/crash) → full reindex of that file
- If mtime changed but size unchanged → full reindex of that file (safety fallback)
- Streaming batch: batch_size=32, process files one at a time (OOM protection for 2GB free RAM)
- Index location: `~/.claude/projects/C--Users-Oleg--local-bin/memory/deja/index.db`
- PID lock file (`~/.claude/projects/C--Users-Oleg--local-bin/memory/deja/index.lock`) — prevents concurrent indexers
- Schema versioning: `meta` table with schema_version, embedding_model, embedding_dim, parser_version
  - On version mismatch → refuse to search, prompt full reindex
- GC orphans: during `deja index`, compare `indexed_files` with filesystem, delete chunks for missing files

### SQLite configuration

```python
PRAGMA journal_mode = WAL;          -- concurrent read/write
PRAGMA synchronous = NORMAL;        -- safe in WAL, faster than FULL
PRAGMA busy_timeout = 5000;         -- 5s wait instead of immediate BUSY
```

- Reader (MCP server): open with `mode=ro` URI parameter
- Checkpoint: `PRAGMA wal_checkpoint(TRUNCATE)` after each completed batch in indexer
- Never use RESTART/FULL checkpoint modes (blocking)

## FTS5 Configuration

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_text,
    tokenize = "unicode61 tokenchars '-._/:'"
);
```

- `tokenchars` keeps `-._/:` as part of tokens: `docker-compose.yml` = one token, not three
- `-` in FTS5 MATCH = NOT operator — always escape user queries:

```python
def fts5_escape(query: str) -> str:
    return '"' + query.replace('"', '""') + '"'
```

- No porter stemmer (incompatible with Russian text)

## Search Strategy

Hybrid search — always run both, combine results:
- **Vector search**: embed query → sqlite-vec KNN (`WHERE embedding MATCH ? AND k = 20`) → top 20
- **FTS5 search**: escape query → FTS5 match → top 20
- **Merge**: Reciprocal Rank Fusion (RRF) → deduplicate → return top N

## MCP Server

- Transport: stdio (default, SSE is deprecated in MCP spec)
- Framework: FastMCP
- Model loading: via lifespan (loads once at startup, not lazily)
- All logging to stderr only — stdout is JSON-RPC protocol, any print() breaks it
- Errors: raise `ToolError` from `fastmcp.exceptions`
- Env: `PYTHONUNBUFFERED=1` required

```python
from fastmcp import FastMCP
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(server):
    model = await asyncio.to_thread(load_embedding_model)
    db = open_index_db()
    yield {"model": model, "db": db}

mcp = FastMCP("deja", lifespan=lifespan)
```

### Tools

#### search

Hybrid semantic + keyword search across all indexed sessions.

- Input:
  - query (string, required)
  - limit (int, default 10)
  - project (string, optional — filter by project path)
  - date_from (string, optional — ISO date)
  - date_to (string, optional — ISO date)
- Process: embed query → vector KNN + FTS5 → RRF merge → apply filters → return top results
- Output: list of chunks with score, timestamp, project_path, session_id, context (user + assistant text)

#### get_session

Read full context of a specific session.

- Input: session_id (string)
- Output: all messages from that session (text only, filtered tool_result noise)

## File Structure

```
C:\Projects\deja\
  src/
    deja/
      __init__.py
      parser.py        # JSONL parser with schema validation
      chunker.py        # text chunking
      indexer.py        # embedding + SQLite storage + GC + locking
      search.py         # hybrid search (vector + FTS5 + RRF)
      server.py         # FastMCP server with lifespan
      db.py             # SQLite schema, migrations, WAL config
      cli.py            # CLI entry point
  pyproject.toml
  README.md
```

## Out of Scope (MVP)

- CLI search (index only via CLI, search only via MCP)
- Other formats (ChatGPT, Gemini, etc.)
- Web UI
- Auto-indexing hooks
- Index encryption / secret filtering
- PyPI publication (later)
- Time decay ranking (later)
- get_context tool with window (later)
- platformdirs for cross-platform paths (later, hardcode for now)

## Performance Targets

- Index 500MB JSONL: < 30 min first run
- Incremental index: < 30 sec
- Search latency (warm): < 500ms
- Search latency (cold, first after startup): < 5 sec (model loading)
- MCP server startup: < 1 sec (model loads in lifespan, first search waits)
- RAM: < 150MB during search, < 300MB during indexing (batch_size=32)
