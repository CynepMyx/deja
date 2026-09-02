# Changelog

## Unreleased

### Features

- **Claude Code sub-agent threads are indexed.** Claude Code writes a separate transcript per delegated thread under `<project>/<session-id>/subagents/agent-*.jsonl`. Discovery only globbed `<project>/*.jsonl`, so all of it was silently invisible — on a workstation that delegates heavily these threads outnumber main sessions, and they hold work the main transcript never contains: delegated research, generated code, sub-agent-only tool calls.
  - New `kind` column on `chunks` and `sessions`, `"main"` or `"subagent"`.
  - `include_subagents` flag on `hybrid_search`, the `search` MCP tool and `deja search`. **Off by default** — a query should answer from the conversation the user actually had; opt in for full recall.
  - Excluding sub-agents widens the candidate pool the same way other filters do, but only when the index actually holds sub-agent chunks — indexes without them keep the cheaper narrow pool.
  - `deja stats` breaks chunk counts down by kind.
  - The exclusion is pushed into both retrieval lanes rather than applied to the merged result. A post-filter let sub-agent chunks consume the candidate slots first: on a 276k-chunk index that is 60% sub-agent, broad FTS queries lost half to three quarters of the keyword lane before the merge.
- **Sub-agent threads link back to the session that spawned them.** `sessions.parent_session_id` is derived from the transcript path, surfaced on every search result, and walkable in the other direction via the new `list_subagent_threads` MCP tool — a delegated finding leads back to the conversation it came from.
- **`deja analytics` is sub-agent aware.** Delegated threads are separate rows in `sessions`; counting them turned "how many sessions did I have" into "how many threads ran" and let delegated work own the per-session rankings. Excluded by default, `--include-subagents` to opt in.
- **`deja stats` prints its session count broken down by kind**, so it no longer silently disagrees with `deja analytics` about what counts as a session.
- **The MCP server survives a stale index.** A schema mismatch was raised inside the lifespan, so the process died at startup and the client only reported that the connection closed. The server now stays up and every tool answers with the version mismatch and the command that fixes it.
- **`deja stats` and `deja search` report a stale index instead of crashing.** Both opened the database without checking its schema version and died on a missing column; `stats` now keeps printing the rest of its diagnostics and flags the mismatch as an issue.

### Schema

- **SCHEMA_VERSION 4 → 5**, applied **in place**: `kind` on `chunks` and `sessions`, `parent_session_id` on `sessions`. Additive versions now migrate with `ALTER TABLE` and keep their embeddings instead of forcing a full re-embed — the previous behaviour cost hours of CPU on a large index for columns that add in milliseconds. Non-additive versions still drop and rebuild. Run `deja index` once afterwards to pick up sub-agent threads.

### API

- `Parser.discover()` now yields `(path, project_path, kind)` triples instead of pairs. Sources without a sub-agent concept yield `"main"`.

### Tests

- 126 total (was 107): discovery, the schema default, chunk tagging, both filter directions, in-place migration with embeddings kept, retrieval-lane filtering under a k smaller than the sub-agent population, analytics scoping, and the parent link in both directions.


## 0.6.0 (2026-08-24)

### Features

- **Third source: `review-extract`** — indexes the markdown digests written by `extract_sessions.py` to `~/.claude/reviews/sessions`. These outlive the transcripts they came from: `cleanupPeriodDays` deletes `.jsonl` files after 30 days, so for early sessions the digest is the only surviving copy. `deja index --source review-extract`, or picked up by `--source all`.
  - `project_path` is encoded the way Claude Code names its project directories and verified against an existing directory under `~/.claude/projects`, so the same project is not counted twice.
  - Timestamps are full ISO — `search.py` feeds them to `fromisoformat` for time decay.
  - Byte offsets tracked per turn, so incremental resume works the same as for the other sources.
  - Files holding only slash commands yield no turns, matching `claude_code` behaviour.

### Fixes

- **Truncated-private-key pattern no longer eats the rest of the input.** The fallback for a `BEGIN` header without a matching `END` ended in `[\s\S]+`. Inside deja that is invisible, because `redact()` only ever sees a 1500-char chunk, but called on larger text it silently deleted everything after the key — on an archived transcript, 6910 lines. The match is now bounded to base64 body characters and stops at the first character that cannot belong to a key.

### Schema

- Unchanged, still SCHEMA_VERSION 4. Upgrading from 0.5.0 does **not** trigger a reindex. Run `deja index --source review-extract` once to pick up the new source.

### Tests

- 107 total (was 105): 13 for the new parser, 2 regression tests for the key pattern.


## 0.5.0 (2026-06-29)

### Features

- **`deja analytics` command (#25)** — read-only usage reports from a new pre-aggregated `sessions` table. Five reports: top by token cost, top by length, by-project, by-tool, by-day sparkline. `--top N`, `--since-days N`, `--format human|json`.
- **Git-state filtering (#24)** — `chunks.git_branch` captured at message time from JSONL records. `deja search --git-branch X` exact match or `--git-branch-prefix feature/`. Same params on MCP `search` tool. Claude Code populates branch from per-record `gitBranch`; Codex from `session_meta.payload.git.branch`; other sources NULL (filter naturally excludes). Note: Claude Code writes the literal string `"HEAD"` instead of the branch name when `cwd` isn't a git checkout or the repo is detached — filtering by `"HEAD"` retrieves those.
- **Parser contract widened** — both `claude-code` and `codex` parsers now emit `usage`, `git_branch`, and `tool_names` per turn. Tokens aggregated across the consecutive assistant entries that make up one turn.
- **Indexer pre-aggregate write path** — every non-provisional turn UPSERTs the `sessions` row (sum tokens, MIN/MAX timestamps, +1 turn_count) and increments per-tool counts in `tool_calls`. `_delete_file_chunks` clears both tables for the session so a full reindex starts from zero.

### Schema (requires reindex)

- **SCHEMA_VERSION 3 → 4.** First run of 0.5.0 detects mismatch and triggers full reindex with a clear log line.
- New tables: `sessions`, `tool_calls`.
- New columns on `chunks`: `git_branch`, `parent_id` (parent_id is currently always NULL — reserved for #13 parent-child chunking in 0.6.0).
- New indexes: `idx_chunks_branch`, `idx_chunks_parent`, `idx_sessions_project`, `idx_sessions_started`.

### Tests

- 92 total (was 82 at 0.4.0) — 10 new for analytics queries and formatting.

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
