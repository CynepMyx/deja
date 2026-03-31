# deja — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MCP server that provides semantic search over Claude Code JSONL session history.

**Architecture:** CLI indexer parses JSONL → builds embeddings → stores in SQLite. MCP server opens pre-built index → answers search/get_session queries via stdio. Hybrid search combines vector KNN + FTS5 via Reciprocal Rank Fusion.

**Tech Stack:** Python 3.10+, fastembed (ONNX), sqlite-vec, SQLite FTS5, FastMCP

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/deja/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "deja"
version = "0.1.0"
description = "Semantic search for Claude Code sessions"
requires-python = ">=3.10"
dependencies = [
    "fastembed>=0.6.0,<1.0",
    "sqlite-vec==0.1.8",
    "fastmcp>=2.0.0,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[project.scripts]
deja = "deja.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/deja"]
```

- [ ] **Step 2: Create package init**

```python
# src/deja/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 3: Create venv and install**

Run:
```bash
cd C:/Projects/deja
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
```

Expected: installs deja + dependencies without errors. fastembed will download ONNX runtime (~50MB).

- [ ] **Step 4: Verify imports**

Run:
```bash
.venv/Scripts/python -c "from fastembed import TextEmbedding; import sqlite_vec; import fastmcp; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Init git repo**

```bash
cd C:/Projects/deja
git init
```

Create `.gitignore`:
```
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore
git commit -m "init: project scaffolding with dependencies"
```

---

## Task 2: SQLite Database Schema

**Files:**
- Create: `src/deja/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing test for DB initialization**

```python
# tests/test_db.py
import os
import tempfile
from deja.db import init_db, get_meta

def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "chunks" in table_names
        assert "chunks_fts" in table_names
        assert "indexed_files" in table_names
        assert "meta" in table_names
        conn.close()

def test_meta_table_has_schema_version():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        meta = get_meta(conn)
        assert meta["schema_version"] == 1
        assert meta["embedding_model"] == "intfloat/multilingual-e5-small"
        assert meta["embedding_dim"] == 384
        conn.close()

def test_wal_mode_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deja.db'`

- [ ] **Step 3: Implement db.py**

```python
# src/deja/db.py
import sqlite3
import struct
import sqlite_vec

SCHEMA_VERSION = 1
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            split_index INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT,
            project_path TEXT,
            chunk_text TEXT NOT NULL,
            tool_result_text TEXT,
            UNIQUE(session_id, message_index, split_index)
        );

        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            last_offset INTEGER NOT NULL DEFAULT 0,
            last_mtime REAL NOT NULL,
            last_size INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding float[384]
        )
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_text,
            tool_result_text,
            tokenize = "unicode61 tokenchars '-._/:'"
        )
    """)

    meta_defaults = {
        "schema_version": str(SCHEMA_VERSION),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": str(EMBEDDING_DIM),
        "parser_version": "1",
    }
    for key, value in meta_defaults.items():
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    conn.commit()
    return conn

def get_meta(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {k: v for k, v in rows}

def open_db_readonly(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/pytest tests/test_db.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/deja/db.py tests/test_db.py
git commit -m "feat: SQLite schema with WAL, FTS5, vec0, meta table"
```

---

## Task 3: JSONL Parser

**Files:**
- Create: `src/deja/parser.py`
- Create: `tests/test_parser.py`
- Create: `tests/fixtures/sample.jsonl`

- [ ] **Step 1: Create test fixture**

```python
# tests/fixtures/sample.jsonl — create via Python to ensure valid JSON
# (will be created in test setup)
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_parser.py
import json
import os
import tempfile
from deja.parser import parse_jsonl_file, extract_content

def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def test_extract_text_content():
    content = [{"type": "text", "text": "Hello world"}]
    text, tool_text = extract_content(content)
    assert text == "Hello world"
    assert tool_text == ""

def test_extract_tool_use():
    content = [
        {"type": "text", "text": "Let me check"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
    ]
    text, tool_text = extract_content(content)
    assert "Let me check" in text
    assert "[Tool: Bash] ls -la" in text

def test_extract_tool_result_separate():
    content = [
        {"type": "tool_result", "content": "total 42\ndrwxr-xr-x 2 user user 4096 file.txt"}
    ]
    text, tool_text = extract_content(content)
    assert text == ""
    assert "total 42" in tool_text

def test_extract_skips_thinking():
    content = [
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "Here is the answer"},
    ]
    text, tool_text = extract_content(content)
    assert "think" not in text.lower()
    assert "Here is the answer" in text

def test_parse_jsonl_file_extracts_turns():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        _write_jsonl(path, [
            {"type": "summary", "summary": "Test session"},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "How to fix nginx?"}]},
                "timestamp": "2026-03-30T10:00:00Z",
                "uuid": "msg-001",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Check the config file."}]},
                "timestamp": "2026-03-30T10:00:05Z",
                "uuid": "msg-002",
            },
        ])
        turns = list(parse_jsonl_file(path))
        assert len(turns) == 1
        assert "nginx" in turns[0]["user_text"]
        assert "config" in turns[0]["assistant_text"]
        assert turns[0]["timestamp"] == "2026-03-30T10:00:05Z"

def test_parse_jsonl_skips_malformed_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broken.jsonl")
        with open(path, "w") as f:
            f.write('{"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"}\n')
        turns = list(parse_jsonl_file(path))
        assert len(turns) == 1

def test_parse_jsonl_with_offset():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        lines = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "first"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp1"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "second"}]}, "timestamp": "2026-01-01T00:00:02Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp2"}]}, "timestamp": "2026-01-01T00:00:03Z", "uuid": "4"},
        ]
        _write_jsonl(path, lines)
        all_turns = list(parse_jsonl_file(path, offset=0))
        assert len(all_turns) == 2
        # Get offset after first two lines
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            f.readline()
            offset = f.tell()
        partial_turns = list(parse_jsonl_file(path, offset=offset))
        assert len(partial_turns) == 1
        assert "second" in partial_turns[0]["user_text"]

def test_tool_result_truncated_to_2000():
    long_result = "x" * 5000
    content = [{"type": "tool_result", "content": long_result}]
    text, tool_text = extract_content(content)
    assert len(tool_text) <= 2000
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deja.parser'`

- [ ] **Step 4: Implement parser.py**

```python
# src/deja/parser.py
import json
import sys
from typing import Generator

TOOL_RESULT_MAX = 2000

def extract_content(content) -> tuple[str, str]:
    if isinstance(content, str):
        return content, ""

    text_parts = []
    tool_result_parts = []

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            if isinstance(inp, dict):
                cmd = inp.get("command", inp.get("file_path", inp.get("query", "")))
            else:
                cmd = str(inp)[:200]
            text_parts.append(f"[Tool: {name}] {cmd}")

        elif block_type == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, list):
                raw = " ".join(
                    b.get("text", "") for b in raw if isinstance(b, dict)
                )
            if isinstance(raw, str):
                tool_result_parts.append(raw[:TOOL_RESULT_MAX])

        elif block_type == "thinking":
            continue

    return "\n".join(text_parts), "\n".join(tool_result_parts)

def parse_jsonl_file(
    path: str, offset: int = 0
) -> Generator[dict, None, None]:
    pending_user = None
    message_index = 0

    with open(path, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[deja] skipping malformed line in {path}", file=sys.stderr)
                continue

            entry_type = entry.get("type", "")

            if entry_type == "summary":
                continue

            message = entry.get("message", {})
            content = message.get("content", [])
            timestamp = entry.get("timestamp", "")

            if entry_type == "user":
                text, tool_text = extract_content(content)
                pending_user = {
                    "text": text,
                    "tool_result": tool_text,
                    "timestamp": timestamp,
                }

            elif entry_type == "assistant" and pending_user is not None:
                text, tool_text = extract_content(content)
                combined_tool = "\n".join(
                    filter(None, [pending_user["tool_result"], tool_text])
                )
                yield {
                    "user_text": pending_user["text"],
                    "assistant_text": text,
                    "tool_result_text": combined_tool[:TOOL_RESULT_MAX],
                    "timestamp": timestamp or pending_user["timestamp"],
                    "message_index": message_index,
                }
                message_index += 1
                pending_user = None

    # Return final file position for offset tracking
    # Caller should use f.tell() after iteration

def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/pytest tests/test_parser.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/deja/parser.py tests/test_parser.py
git commit -m "feat: JSONL parser with offset support, tool_result separation"
```

---

## Task 4: Chunker

**Files:**
- Create: `src/deja/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunker.py
from deja.chunker import make_chunks

def test_short_turn_single_chunk():
    turn = {
        "user_text": "How to restart nginx?",
        "assistant_text": "Run: sudo systemctl restart nginx",
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert len(chunks) == 1
    assert "nginx" in chunks[0]["chunk_text"]
    assert chunks[0]["session_id"] == "sess-1"
    assert chunks[0]["split_index"] == 0

def test_long_turn_splits():
    turn = {
        "user_text": "Explain everything about Docker containers.",
        "assistant_text": "A" * 2000,
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["split_index"] == i
        assert len(chunk["chunk_text"]) <= 1700  # 1500 + some overhead

def test_tool_result_not_in_chunk_text():
    turn = {
        "user_text": "Check disk",
        "assistant_text": "Here are the results",
        "tool_result_text": "Filesystem Size Used Avail",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert "Filesystem" not in chunks[0]["chunk_text"]
    assert chunks[0]["tool_result_text"] == "Filesystem Size Used Avail"

def test_split_respects_sentence_boundaries():
    sentences = ". ".join([f"Sentence number {i}" for i in range(50)])
    turn = {
        "user_text": "Tell me many things",
        "assistant_text": sentences,
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    for chunk in chunks:
        text = chunk["chunk_text"]
        if not text.startswith("Tell me"):
            assert not text[0].islower() or text.startswith(". ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement chunker.py**

```python
# src/deja/chunker.py
MAX_CHUNK_SIZE = 1500
OVERLAP = 200

def _split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Find sentence boundary
        search_region = text[end - OVERLAP:end]
        for sep in ["\n\n", ". ", ".\n", "\n"]:
            pos = search_region.rfind(sep)
            if pos != -1:
                end = end - OVERLAP + pos + len(sep)
                break

        chunks.append(text[start:end])
        start = end - OVERLAP
        if start < 0:
            start = 0

    return chunks

def make_chunks(
    turn: dict, session_id: str, project_path: str
) -> list[dict]:
    embed_text = f"{turn['user_text']}\n\n{turn['assistant_text']}"
    parts = _split_text(embed_text)

    return [
        {
            "chunk_text": part,
            "tool_result_text": turn.get("tool_result_text", "") if i == 0 else "",
            "session_id": session_id,
            "message_index": turn["message_index"],
            "split_index": i,
            "timestamp": turn.get("timestamp", ""),
            "project_path": project_path,
        }
        for i, part in enumerate(parts)
    ]
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/pytest tests/test_chunker.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/deja/chunker.py tests/test_chunker.py
git commit -m "feat: chunker with sentence-boundary splitting"
```

---

## Task 5: Indexer (Embedding + Storage)

**Files:**
- Create: `src/deja/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_indexer.py
import json
import os
import tempfile
from deja.db import init_db
from deja.indexer import index_file, get_embedding_model, check_needs_reindex

def _make_session(tmp, filename, lines):
    path = os.path.join(tmp, filename)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path

def test_index_file_inserts_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "test-project")
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count >= 1
        vec_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        assert vec_count >= 1
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_count >= 1
        conn.close()

def test_incremental_index_skips_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "test"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        count1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        needs = check_needs_reindex(conn, path)
        assert needs == False
        conn.close()

def test_safe_reindex_on_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "test"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        # Simulate truncation
        conn.execute(
            "UPDATE indexed_files SET last_offset = 99999 WHERE path = ?",
            (path,),
        )
        conn.commit()
        needs = check_needs_reindex(conn, path)
        assert needs == "full"
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_indexer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement indexer.py**

```python
# src/deja/indexer.py
import os
import sys
from fastembed import TextEmbedding
from fastembed.text.text_embedding import PoolingType, ModelSource
from deja.db import serialize_f32
from deja.parser import parse_jsonl_file, get_file_end_offset
from deja.chunker import make_chunks

BATCH_SIZE = 32

def get_embedding_model() -> TextEmbedding:
    TextEmbedding.add_custom_model(
        model="intfloat/multilingual-e5-small",
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf="intfloat/multilingual-e5-small"),
        dim=384,
        model_file="onnx/model.onnx",
    )
    return TextEmbedding(model_name="intfloat/multilingual-e5-small")

def check_needs_reindex(conn, path: str) -> bool | str:
    row = conn.execute(
        "SELECT last_offset, last_mtime, last_size FROM indexed_files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return "full"

    last_offset, last_mtime, last_size = row
    stat = os.stat(path)
    current_size = stat.st_size
    current_mtime = stat.st_mtime

    if current_size < last_offset:
        return "full"
    if current_mtime != last_mtime and current_size == last_size:
        return "full"
    if current_mtime == last_mtime and current_size == last_size:
        return False
    return "incremental"

def _delete_file_chunks(conn, session_id: str):
    chunk_ids = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ?", (session_id,)
    ).fetchall()
    for (cid,) in chunk_ids:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))

def index_file(conn, model: TextEmbedding, path: str, project_path: str):
    session_id = os.path.splitext(os.path.basename(path))[0]
    needs = check_needs_reindex(conn, path)

    if needs == False:
        return

    offset = 0
    if needs == "full":
        _delete_file_chunks(conn, session_id)
    elif needs == "incremental":
        row = conn.execute(
            "SELECT last_offset FROM indexed_files WHERE path = ?", (path,)
        ).fetchone()
        offset = row[0] if row else 0

    turns = list(parse_jsonl_file(path, offset=offset))
    if not turns:
        _update_file_meta(conn, path)
        return

    all_chunks = []
    for turn in turns:
        all_chunks.extend(make_chunks(turn, session_id, project_path))

    # Batch embed
    texts = [c["chunk_text"] for c in all_chunks]
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        batch_chunks = all_chunks[batch_start:batch_start + BATCH_SIZE]
        embeddings = list(model.embed(batch_texts))

        for chunk, embedding in zip(batch_chunks, embeddings):
            try:
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO chunks
                    (session_id, message_index, split_index, timestamp, project_path, chunk_text, tool_result_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk["session_id"], chunk["message_index"], chunk["split_index"],
                     chunk["timestamp"], chunk["project_path"], chunk["chunk_text"],
                     chunk.get("tool_result_text", "")),
                )
                rowid = cursor.lastrowid
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, serialize_f32(embedding.tolist())),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                    (rowid, chunk["chunk_text"], chunk.get("tool_result_text", "")),
                )
            except Exception as e:
                print(f"[deja] error inserting chunk: {e}", file=sys.stderr)

    _update_file_meta(conn, path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()

def _update_file_meta(conn, path: str):
    stat = os.stat(path)
    conn.execute(
        """INSERT OR REPLACE INTO indexed_files (path, last_offset, last_mtime, last_size)
        VALUES (?, ?, ?, ?)""",
        (path, stat.st_size, stat.st_mtime, stat.st_size),
    )

def gc_orphans(conn, known_paths: set[str]):
    indexed = conn.execute("SELECT path FROM indexed_files").fetchall()
    for (path,) in indexed:
        if path not in known_paths:
            session_id = os.path.splitext(os.path.basename(path))[0]
            _delete_file_chunks(conn, session_id)
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (path,))
            print(f"[deja] gc: removed orphan {path}", file=sys.stderr)
    conn.commit()
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/pytest tests/test_indexer.py -v`
Expected: all PASS (first run will download model ~117MB)

- [ ] **Step 5: Commit**

```bash
git add src/deja/indexer.py tests/test_indexer.py
git commit -m "feat: indexer with incremental/safe reindex, GC, batch embedding"
```

---

## Task 6: Hybrid Search

**Files:**
- Create: `src/deja/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import json
import os
import tempfile
from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.search import hybrid_search, fts5_escape

def _make_indexed_db(tmp):
    db_path = os.path.join(tmp, "test.db")
    conn = init_db(db_path)
    model = get_embedding_model()
    path = os.path.join(tmp, "session.jsonl")
    lines = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Как настроить nginx reverse proxy?"}]}, "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Создайте файл /etc/nginx/sites-available/proxy.conf с proxy_pass директивой"}]}, "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
        {"type": "user", "message": {"content": [{"type": "text", "text": "Как создать docker-compose.yml?"}]}, "timestamp": "2026-03-30T11:00:00Z", "uuid": "3"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Создайте файл docker-compose.yml в корне проекта"}]}, "timestamp": "2026-03-30T11:00:05Z", "uuid": "4"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    index_file(conn, model, path, "test-project")
    return conn, model

def test_search_finds_relevant():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_indexed_db(tmp)
        results = hybrid_search(conn, model, "nginx proxy", limit=5)
        assert len(results) > 0
        assert "nginx" in results[0]["chunk_text"].lower()
        conn.close()

def test_search_with_project_filter():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_indexed_db(tmp)
        results = hybrid_search(conn, model, "nginx", limit=5, project="test-project")
        assert len(results) > 0
        results_none = hybrid_search(conn, model, "nginx", limit=5, project="nonexistent")
        assert len(results_none) == 0
        conn.close()

def test_fts5_escape_handles_special_chars():
    assert fts5_escape('docker-compose') == '"docker-compose"'
    assert fts5_escape('test "quotes"') == '"test ""quotes"""'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_search.py -v`
Expected: FAIL

- [ ] **Step 3: Implement search.py**

```python
# src/deja/search.py
import sqlite3
from deja.db import serialize_f32

def fts5_escape(query: str) -> str:
    return '"' + query.replace('"', '""') + '"'

def _vector_search(conn, model, query: str, k: int = 20) -> list[dict]:
    query_embedding = list(model.embed([query]))[0]
    rows = conn.execute(
        """
        WITH vec_results AS (
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        )
        SELECT c.id, c.session_id, c.message_index, c.timestamp,
               c.project_path, c.chunk_text, c.tool_result_text,
               v.distance
        FROM vec_results v
        JOIN chunks c ON c.id = v.rowid
        """,
        (serialize_f32(query_embedding.tolist()), k),
    ).fetchall()

    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "distance": r[7],
        }
        for r in rows
    ]

def _fts_search(conn, query: str, k: int = 20) -> list[dict]:
    escaped = fts5_escape(query)
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.session_id, c.message_index, c.timestamp,
                   c.project_path, c.chunk_text, c.tool_result_text,
                   rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (escaped, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "fts_rank": r[7],
        }
        for r in rows
    ]

def _rrf_merge(vec_results: list, fts_results: list, k: int = 60) -> list[dict]:
    scores = {}
    items = {}

    for rank, item in enumerate(vec_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    for rank, item in enumerate(fts_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in items:
            items[doc_id] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        item = items[doc_id]
        item["score"] = scores[doc_id]
        results.append(item)

    return results

def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
) -> list[dict]:
    vec_results = _vector_search(conn, model, query, k=20)
    fts_results = _fts_search(conn, query, k=20)
    merged = _rrf_merge(vec_results, fts_results)

    if project:
        merged = [r for r in merged if r.get("project_path") == project]
    if date_from:
        merged = [r for r in merged if r.get("timestamp", "") >= date_from]
    if date_to:
        merged = [r for r in merged if r.get("timestamp", "") <= date_to]

    return merged[:limit]
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/pytest tests/test_search.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/deja/search.py tests/test_search.py
git commit -m "feat: hybrid search with vector + FTS5 + RRF merge"
```

---

## Task 7: MCP Server

**Files:**
- Create: `src/deja/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_server.py
import json
import os
import tempfile
from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.server import _do_search, _do_get_session

def _make_test_index(tmp):
    db_path = os.path.join(tmp, "test.db")
    conn = init_db(db_path)
    model = get_embedding_model()
    path = os.path.join(tmp, "sess-001.jsonl")
    lines = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Fix SSL certificate"}]}, "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Run certbot renew"}]}, "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    index_file(conn, model, path, "test-project")
    return conn, model

def test_do_search_returns_results():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_test_index(tmp)
        result = _do_search(conn, model, "SSL certificate", limit=5)
        assert len(result) > 0
        assert "session_id" in result[0]
        conn.close()

def test_do_get_session():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_test_index(tmp)
        result = _do_get_session(conn, "sess-001")
        assert len(result) > 0
        assert "SSL" in result[0]["chunk_text"] or "certbot" in result[0]["chunk_text"]
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest tests/test_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement server.py**

```python
# src/deja/server.py
import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from deja.db import open_db_readonly, get_meta, SCHEMA_VERSION
from deja.indexer import get_embedding_model
from deja.search import hybrid_search

DEFAULT_INDEX_PATH = os.path.join(
    os.path.expanduser("~"),
    ".claude", "projects", "C--Users-Oleg--local-bin", "memory", "deja", "index.db",
)

def _check_schema(conn):
    meta = get_meta(conn)
    db_version = int(meta.get("schema_version", "0"))
    if db_version != SCHEMA_VERSION:
        raise ToolError(
            f"Index schema version mismatch: expected {SCHEMA_VERSION}, got {db_version}. "
            "Run 'deja index --reindex' to rebuild."
        )

@asynccontextmanager
async def lifespan(server):
    index_path = os.environ.get("DEJA_INDEX_PATH", DEFAULT_INDEX_PATH)
    if not os.path.exists(index_path):
        print(f"[deja] index not found at {index_path}, search will fail", file=sys.stderr)
        yield {"model": None, "db": None}
        return

    print(f"[deja] loading model...", file=sys.stderr)
    model = await asyncio.to_thread(get_embedding_model)
    db = open_db_readonly(index_path)
    _check_schema(db)
    print(f"[deja] ready", file=sys.stderr)
    yield {"model": model, "db": db}
    db.close()

mcp = FastMCP("deja", lifespan=lifespan)

def _do_search(conn, model, query, limit=10, project=None, date_from=None, date_to=None):
    return hybrid_search(conn, model, query, limit=limit,
                         project=project, date_from=date_from, date_to=date_to)

def _do_get_session(conn, session_id):
    rows = conn.execute(
        """SELECT chunk_text, message_index, timestamp, project_path
        FROM chunks WHERE session_id = ? ORDER BY message_index, split_index""",
        (session_id,),
    ).fetchall()
    return [
        {"chunk_text": r[0], "message_index": r[1], "timestamp": r[2], "project_path": r[3]}
        for r in rows
    ]

@mcp.tool()
async def search(
    query: str,
    limit: int = 10,
    project: str = None,
    date_from: str = None,
    date_to: str = None,
    ctx=None,
) -> list[dict]:
    """Search past Claude Code sessions by meaning. Returns relevant conversation chunks with context."""
    lifespan_ctx = ctx.request_context.lifespan_context
    model = lifespan_ctx.get("model")
    db = lifespan_ctx.get("db")
    if model is None or db is None:
        raise ToolError("Index not loaded. Run 'deja index' first.")
    return await asyncio.to_thread(_do_search, db, model, query, limit, project, date_from, date_to)

@mcp.tool()
async def get_session(session_id: str, ctx=None) -> list[dict]:
    """Get full context of a specific session by session_id."""
    lifespan_ctx = ctx.request_context.lifespan_context
    db = lifespan_ctx.get("db")
    if db is None:
        raise ToolError("Index not loaded. Run 'deja index' first.")
    results = await asyncio.to_thread(_do_get_session, db, session_id)
    if not results:
        raise ToolError(f"Session '{session_id}' not found in index.")
    return results
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/pytest tests/test_server.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/deja/server.py tests/test_server.py
git commit -m "feat: MCP server with search and get_session tools"
```

---

## Task 8: CLI Entry Point

**Files:**
- Create: `src/deja/cli.py`

- [ ] **Step 1: Implement cli.py**

```python
# src/deja/cli.py
import os
import sys
import glob
import fcntl  # Unix; on Windows use msvcrt
import argparse

from deja.db import init_db, get_meta, SCHEMA_VERSION
from deja.indexer import get_embedding_model, index_file, gc_orphans

DEFAULT_INDEX_DIR = os.path.join(
    os.path.expanduser("~"),
    ".claude", "projects", "C--Users-Oleg--local-bin", "memory", "deja",
)
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_INDEX_DIR, "index.db")
LOCK_PATH = os.path.join(DEFAULT_INDEX_DIR, "index.lock")

CLAUDE_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".claude", "projects")

def _acquire_lock():
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    lock_fd = open(LOCK_PATH, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("[deja] another indexer is running, exiting", file=sys.stderr)
        sys.exit(1)
    return lock_fd

def _release_lock(lock_fd):
    lock_fd.close()
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass

def _find_jsonl_files() -> list[tuple[str, str]]:
    results = []
    if not os.path.isdir(CLAUDE_PROJECTS_DIR):
        print(f"[deja] {CLAUDE_PROJECTS_DIR} not found", file=sys.stderr)
        return results

    for project_dir in os.listdir(CLAUDE_PROJECTS_DIR):
        full_project = os.path.join(CLAUDE_PROJECTS_DIR, project_dir)
        if not os.path.isdir(full_project):
            continue
        for jsonl in glob.glob(os.path.join(full_project, "*.jsonl")):
            results.append((jsonl, project_dir))

    return results

def cmd_index(args):
    lock_fd = _acquire_lock()
    try:
        os.makedirs(DEFAULT_INDEX_DIR, exist_ok=True)
        conn = init_db(DEFAULT_INDEX_PATH)

        meta = get_meta(conn)
        if args.reindex or int(meta.get("schema_version", "0")) != SCHEMA_VERSION:
            print("[deja] full reindex requested", file=sys.stderr)
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_vec")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM indexed_files")
            conn.commit()

        print("[deja] loading embedding model...", file=sys.stderr)
        model = get_embedding_model()

        files = _find_jsonl_files()
        print(f"[deja] found {len(files)} JSONL files", file=sys.stderr)

        known_paths = set()
        for i, (path, project) in enumerate(files):
            known_paths.add(path)
            print(f"[deja] [{i+1}/{len(files)}] {os.path.basename(path)}", file=sys.stderr)
            index_file(conn, model, path, project)

        gc_orphans(conn, known_paths)
        conn.close()
        print("[deja] indexing complete", file=sys.stderr)
    finally:
        _release_lock(lock_fd)

def cmd_serve(args):
    from deja.server import mcp
    mcp.run(transport="stdio")

def main():
    parser = argparse.ArgumentParser(prog="deja", description="Semantic search for Claude Code sessions")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index JSONL session files")
    idx.add_argument("--reindex", action="store_true", help="Force full reindex")

    sub.add_parser("serve", help="Start MCP server (stdio)")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test CLI manually**

Run:
```bash
cd C:/Projects/deja
.venv/Scripts/deja index
```

Expected: scans `~/.claude/projects/*/`, indexes JSONL files, prints progress.

- [ ] **Step 3: Test serve mode**

Run:
```bash
.venv/Scripts/deja serve
```

Expected: starts stdio MCP server, waits for input (Ctrl+C to exit).

- [ ] **Step 4: Fix Windows locking**

The `fcntl` import will fail on Windows. The code already has a `sys.platform == "win32"` branch with `msvcrt`. Remove the top-level `import fcntl` and make it conditional:

```python
# Replace top-level import fcntl with:
# (no top-level import — handled in _acquire_lock)
```

- [ ] **Step 5: Commit**

```bash
git add src/deja/cli.py
git commit -m "feat: CLI with index and serve commands, PID locking"
```

---

## Task 9: Integration Test + MCP Config

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import json
import os
import tempfile
from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.search import hybrid_search

def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "index.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        # Create two sessions
        for sess_name, turns in [
            ("nginx-session", [
                ("Как настроить SSL на nginx?", "Используйте certbot: sudo certbot --nginx -d example.com"),
                ("А как проверить сертификат?", "Выполните: openssl s_client -connect example.com:443"),
            ]),
            ("docker-session", [
                ("Как запустить docker-compose?", "Выполните: docker-compose up -d в директории с docker-compose.yml"),
            ]),
        ]:
            path = os.path.join(tmp, f"{sess_name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i, (user, assistant) in enumerate(turns):
                    f.write(json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": user}]}, "timestamp": f"2026-03-30T{10+i}:00:00Z", "uuid": f"u{i}"}, ensure_ascii=False) + "\n")
                    f.write(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": assistant}]}, "timestamp": f"2026-03-30T{10+i}:00:05Z", "uuid": f"a{i}"}, ensure_ascii=False) + "\n")
            index_file(conn, model, path, "test")

        # Search in Russian
        results = hybrid_search(conn, model, "SSL сертификат nginx", limit=5)
        assert len(results) > 0
        top = results[0]
        assert "nginx" in top["chunk_text"].lower() or "ssl" in top["chunk_text"].lower() or "certbot" in top["chunk_text"].lower()

        # Search for docker
        results = hybrid_search(conn, model, "docker-compose запуск", limit=5)
        assert len(results) > 0
        assert "docker" in results[0]["chunk_text"].lower()

        conn.close()
```

- [ ] **Step 2: Run integration test**

Run: `.venv/Scripts/pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Add MCP config to ~/.claude.json**

Add to `mcpServers` in `~/.claude.json`:
```json
"deja": {
    "type": "stdio",
    "command": "C:/Projects/deja/.venv/Scripts/deja.exe",
    "args": ["serve"],
    "env": {
        "PYTHONUNBUFFERED": "1"
    }
}
```

- [ ] **Step 4: Run all tests**

Run: `.venv/Scripts/pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration test, MCP config ready"
```

---

## Task 10: First Real Index + Smoke Test

- [ ] **Step 1: Run first real indexation**

```bash
cd C:/Projects/deja
.venv/Scripts/deja index
```

Expected: indexes all 73+ sessions from `~/.claude/projects/*/`. Takes 10-30 min on first run.

- [ ] **Step 2: Check index stats**

```bash
.venv/Scripts/python -c "
import sqlite3, sqlite_vec
conn = sqlite3.connect(os.path.expanduser('~/.claude/projects/C--Users-Oleg--local-bin/memory/deja/index.db'))
conn.enable_load_extension(True)
sqlite_vec.load(conn)
print('Chunks:', conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0])
print('Vectors:', conn.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()[0])
print('Files:', conn.execute('SELECT COUNT(*) FROM indexed_files').fetchone()[0])
"
```

- [ ] **Step 3: Restart Claude Code, verify MCP connects**

Run `/mcp` — deja should appear as connected MCP server.

- [ ] **Step 4: Test search from Claude Code**

Ask Claude to search: "как мы чинили SSL" — should return relevant chunks from past sessions.

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "v0.1.0: deja MVP complete"
```
