# Changelog

## 0.4.0 (2026-06-29)

### Features
- **Multi-source support (#8)** — index sessions from both Claude Code and OpenAI Codex CLI in one unified database. `~/.codex/sessions/YYYY/MM/DD/*.jsonl` is walked directly (session_index.jsonl is ignored — it drifts)
- **Parser registry** — adding Cursor / Gemini / OpenCode parsers no longer touches CLI or indexer; each parser module declares `SOURCE`, `discover()`, `parse()`
- **`--source` filter** — on `deja index`, `deja search` and the MCP `search` tool (values: `all`, `claude-code`, `codex`)
- **Source in search results** — every result carries a `source` field

### Bug fixes
- **`gc_orphans` scoped to indexed sources** — `deja index --source codex` no longer treats every claude-code file as orphan and wipes its chunks
- **Provisional EOF turn** — last turn of a growing JSONL is now re-indexed on the next run instead of frozen at past-EOF offset
- **Pre-parse stat in file meta** — data appended during indexing is no longer stamped as already indexed
- **Dangling user at offset=0** — `offset=0` is no longer treated as falsy, so the first turn of a partially-indexed session isn't skipped forever
- **Consecutive assistant entries** — Claude Code splits one model turn into several assistant JSONL entries; the old pairing kept only the first and dropped the rest. Backports the codex accumulate-until-next-user strategy

### Secrets redaction
- **Redact at turn level, before truncate and chunk** — private keys split across the 2000-char tool-result boundary or a chunk split are now caught; embeddings are computed over redacted text
- **Bare high-entropy tokens** — sk-ant/sk-proj, DigitalOcean, Figma, npm, PyPI, standalone JWTs, Google API keys, Stripe live keys and Telegram bot tokens are now caught without assignment context (the common case in tool results)
- **Pattern fixes** — Telegram bot IDs of any digit length, TestPyPI tokens, DO OAuth/refresh siblings; `npm_package_*` / `npm_config_*` env vars no longer false-positive

### Schema (requires reindex)
- **SCHEMA_VERSION 1 → 3.** First run of 0.4.0 detects the mismatch and triggers a full reindex automatically with a clear log line
- New columns: `chunks.source`, `indexed_files.source`, `indexed_files.next_message_index`
- New index: `idx_chunks_source`
- `_migrate_if_needed` in `init_db` — drops old index tables on version mismatch instead of crashing on `CREATE INDEX`

### Tests
- 82 total (was 41 at 0.3.0)

## 0.3.0 (2026-04-02)

### Features
- **`get_context` MCP tool** — retrieve a chunk with surrounding turns (±window) from the same session (#4)
- **Secret filtering** — passwords, API keys, tokens, and private keys are redacted during indexing (#5)
- **`deja redact`** — update secrets in existing index without re-embedding (seconds vs full reindex)
- **`platformdirs`** — index stored in OS-standard location; auto-detects legacy `~/.claude/deja/` (#7)

### Tests
- Tests for get_context, secret redaction, redact command, stats health check (41 total)

### CI
- Upgraded to Node24 actions (checkout@v5, setup-python@v6, cache@v5)

## 0.2.0 (2026-04-02)

### Breaking Changes
- `get_session` MCP tool renamed to `get_session_chunks` (honest about returning chunks, not raw messages)
- Requires FastMCP >= 3.0.0 (upgraded from 2.x)

### Bug Fixes
- **Incremental indexing correctness (P0)** — message_index no longer collides on incremental runs; dangling user at offset boundary no longer lost; stable upsert preserves rowid (#16)
- **Streaming indexer** — process turns in batches (TURNS_PER_BATCH=50) instead of loading entire file into memory (#9). Note: fastembed + ONNX runtime still uses ~3GB RAM for the model itself
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
