# Changelog

## Unreleased

### Features

- **`deja eval` can judge results by chunk, not only by session.** A golden pair may now set `expected_text` (strings the right chunk contains in `chunk_text` or `tool_result_text`), `expected_date_from` / `expected_date_to` (the day range of the original conversation) and `filters` (passed to the search). All criteria must hold for the same result. Pairs with only `expected_sessions` work as before. Session-level pairs say little when a few long sessions hold most of the index: on a real one, 5 of 234 sessions hold 83% of chunks.

### Fixes

- **Filters narrow the candidates before ranking.** `hybrid_search` used to take the global top 20 (top 100 with filters) from the whole index and only then drop rows outside `project` / `source` / `git_branch` / dates. On a 165k-chunk index a one-day filter usually came back empty although that day had hundreds of chunks. Both legs now filter first: FTS gets the conditions in its `WHERE`, the vector leg ranks exact L2 distances over the matching ids when there are at most 20 000 of them, and for wider filters runs KNN with `k` scaled by the filter's selectivity (capped at 4096). On a copy of a real index: a file name searched within one day went from 0 to 5 hits, narrow searches got faster (~0.3 s vs ~0.5 s), unfiltered and wide-filter searches are unchanged.
- **A date-only `date_to` includes that day.** Timestamps are stored as `2026-08-02T16:49:32Z`, so `date_to="2026-08-02"` used to exclude the whole of 2 August, and `date_from == date_to` always returned nothing. A full timestamp is still compared as given.
- **`git_branch_prefix` is matched literally**, so `_` and `%` in a prefix are not wildcards.
- **File names and path fragments are findable.** The FTS tokenizer was `unicode61 tokenchars '-._/:'`, which kept a whole path such as `/opt/app/scripts/monitor.sh.bak-1` as one token: the file name alone, or `scripts/monitor.sh`, never matched. Plain `unicode61` splits on punctuation, and since every query token is quoted it matches as a phrase, so `monitor.sh.bak-1`, `scripts/monitor.sh` and the full path all find it. IPs such as `77.237.235.33` still match as phrases. Query tokens with no word character (`/` in `rm -rf /`, `-` in `cd -`) are dropped: the new tokenizer yields nothing for them, and an empty phrase made the whole AND match no rows.

- **Tool results land on the turn that called the tool** (Claude Code parser). A turn starts at every `user` record, and Claude Code writes tool results as `user` records, so each result was stored on the next step, next to a different command; when several results came in a row, all but the last were dropped. Checked against all 152 local sessions (128 921 turns): before, 84 066 of 94 987 checkable results sat on the wrong turn; now 128, all of them results whose text is masked by secret redaction, so the check cannot see them. Stored tool text grew from 81.2M to 86.0M characters.
  - Finished turns wait in a queue until each `tool_use` has its `tool_result`, matched by `tool_use_id`; parallel calls that Claude Code writes interleaved with their results (call, call, result, call, result) are handled. A turn is released early when an assistant entry of a different `message.id` starts, since by then the API has all results.
  - Turn boundaries, `user_text`, `assistant_text` and `message_index` are byte-identical to before (same 152 sessions), so embeddings stay valid.
  - Turns still waiting for results at EOF are provisional and the next run restarts from the first of them (`resume_message_index` on the turn).
  - Codex is unaffected: its parser already keeps calls and outputs in one turn up to the next user message.

- **A run killed mid-file no longer leaves the file marked as fully indexed.** `index_file` commits after every batch of 50 turns, and every commit stored the file's final size and mtime. If the run died after the first batch (hook killed, app closed, embedding error), `check_needs_reindex` saw an unchanged size and mtime and skipped the rest of the file for good. On a real index one 22 MB session held turns 0–49 of 1695 for months. Per-batch rows now store `last_size = -1` (partial), and the real size and mtime are written only once the whole file is parsed, so the next run resumes from the last committed batch.

### Upgrade

- Still SCHEMA_VERSION 4, embeddings are kept. The first `deja index` after upgrading notices the old tokenizer and rebuilds only the FTS table from `chunks` in one transaction (about 25 s for 165k chunks); if interrupted, the old table stays and the next run retries. Until then search keeps the old tokenizer behaviour.
- Indexes built by parser v1 (`meta.parser_version = 1`) get their tool results reattached on the first `deja index`: every Claude Code file is re-parsed and only `tool_result_text` and its FTS row are rewritten, embeddings kept, one commit per file, the version bumped at the end. A session whose `chunk_text` does not match is reindexed in full. On a copy of a 165k-chunk index: 119 103 chunks updated, 27 307 already right, 0 sessions reindexed, 4.4 minutes at normal priority. New indexes start at version 2.
- Files left behind by that bug in older versions are found once, on the first `deja index`: each file whose stored offset is short of its size is parsed from that offset, and it is marked partial if a complete (non-provisional) turn is still there. Normally that tail is just the last provisional turn: 225 files took 0.3 s.

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
