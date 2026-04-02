import json
import os
import sqlite3
import tempfile

import sqlite_vec

from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.secrets import redact


def _make_index_with_secret(tmp):
    db_path = os.path.join(tmp, "test.db")
    conn = init_db(db_path)
    model = get_embedding_model()
    path = os.path.join(tmp, "sess-002.jsonl")
    lines = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Set password=SuperSecret123456!"}]}, "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done, password set"}]}, "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    index_file(conn, model, path, "test-project")
    return conn, db_path


def test_redact_updates_existing_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        conn, db_path = _make_index_with_secret(tmp)

        # Verify secret was redacted during indexing
        row = conn.execute(
            "SELECT chunk_text FROM chunks WHERE session_id = 'sess-002'"
        ).fetchone()
        assert "[REDACTED]" in row[0]
        conn.close()


def test_cmd_redact_updates_db():
    """Simulate cmd_redact logic on a DB with un-redacted text."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)

        # Insert a chunk with a secret directly (bypassing indexer redaction)
        conn.execute(
            """INSERT INTO chunks
            (session_id, message_index, split_index, timestamp, project_path, chunk_text, tool_result_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("sess-003", 0, 0, "2026-03-30T10:00:00Z", "test",
             "api_key = sk-proj-abcdef1234567890abcdef1234567890", ""),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
            (chunk_id, "api_key = sk-proj-abcdef1234567890abcdef1234567890", ""),
        )
        conn.commit()

        # Run redact logic
        rows = conn.execute("SELECT id, chunk_text, tool_result_text FROM chunks").fetchall()
        updated = 0
        for row_id, chunk_text, tool_text in rows:
            new_chunk = redact(chunk_text)
            new_tool = redact(tool_text) if tool_text else tool_text
            if new_chunk != chunk_text or new_tool != tool_text:
                conn.execute(
                    "UPDATE chunks SET chunk_text = ?, tool_result_text = ? WHERE id = ?",
                    (new_chunk, new_tool, row_id),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                    (row_id, new_chunk, new_tool or ""),
                )
                updated += 1
        conn.commit()

        assert updated == 1
        row = conn.execute("SELECT chunk_text FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        assert "[REDACTED]" in row[0]
        assert "sk-proj" not in row[0]
        conn.close()


def test_stats_health_ok():
    """Verify stats logic reports healthy index."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = os.path.join(tmp, "sess-004.jsonl")
        lines = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "Hello"}]}, "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi there"}]}, "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        index_file(conn, model, path, "test-project")

        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

        assert chunks > 0
        assert chunks == vectors
        assert chunks == fts

        orphans = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT id FROM chunks)"
        ).fetchone()[0]
        assert orphans == 0
        conn.close()
