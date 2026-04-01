# Changelog

## 0.3.0 (unreleased)

### Features
- **`get_context` MCP tool** — retrieve a chunk with surrounding turns (±window) from the same session; no need to fetch entire session (#4)
- **Secret filtering** — passwords, API keys, tokens, and private keys are redacted (`[REDACTED]`) during indexing (#5)

### Tests
- Added tests for get_context (window, not found)
- Added tests for secret redaction (AWS, GitHub, Bearer, passwords, private keys)

## 0.2.0 (2026-04-02)

### Breaking Changes
- `get_session` MCP tool renamed to `get_session_chunks` (honest about returning chunks, not raw messages)
- Requires FastMCP >= 3.0.0 (upgraded from 2.x)

### Bug Fixes
- **Incremental indexing correctness (P0)** — message_index no longer collides on incremental runs; dangling user at offset boundary no longer lost; stable upsert preserves rowid (#16)
- **Streaming indexer** — process turns in batches (TURNS_PER_BATCH=50) instead of loading entire file into memory; ~50MB RAM instead of ~4GB for 86MB files (#9)
- **FTS query** — token-wise AND instead of exact phrase match; `nginx proxy` now finds results with both words in any order (#18)
- **Search filters** — overfetch candidates (k=100) when project/date filters are active (#19)
- **FastMCP upgrade** — use public `ctx.lifespan_context` API instead of private `_lifespan_result` (#10)
- **SQLite threading** — `check_same_thread=False` for async MCP server
- **Windows UTF-8** — force UTF-8 stdout/stderr on Windows to prevent cp1252 crashes on Cyrillic
- **Hardcoded paths** — index location changed to `~/.claude/deja/` (no user-specific paths in code)

### Features
- `deja eval` command with MRR@5 scoring for search quality benchmarking
- Time decay scoring (disabled by default, `time_decay=True` to enable)
- Auto-indexing docs in README (Claude Code Stop hook)
- SQLite indexes on `session_id` and `(project_path, timestamp)`

### Repo
- GitHub Actions CI (ubuntu + windows, Python 3.10 + 3.13)
- README with badges, architecture diagram, usage docs
- CONTRIBUTING.md, .editorconfig, LICENSE (MIT)
- Social preview image

## 0.1.0 (2026-03-31)

Initial MVP release.

- JSONL parser with offset support and tool_result separation
- Chunker with sentence-boundary splitting (1500 chars, 200 overlap)
- Indexer with incremental/safe reindex, GC, batch embedding
- Hybrid search: vector KNN + FTS5 + Reciprocal Rank Fusion
- MCP server (FastMCP, stdio transport) with `search` and `get_session` tools
- CLI: `deja index`, `deja serve`
- fastembed (multilingual-e5-small, 384-dim ONNX) + sqlite-vec + SQLite FTS5
