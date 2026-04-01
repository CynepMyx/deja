import json
import os
import time
import tempfile
from deja.db import init_db
from deja.indexer import index_file, get_embedding_model, check_needs_reindex

def _make_session(tmp, filename, lines):
    path = os.path.join(tmp, filename)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path

def _append_lines(path, lines):
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

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
        conn.execute(
            "UPDATE indexed_files SET last_offset = 99999 WHERE path = ?",
            (path,),
        )
        conn.commit()
        needs = check_needs_reindex(conn, path)
        assert needs == "full"
        conn.close()

def test_incremental_append_no_collision():
    """New turns appended to file get correct message_index, not colliding with old ones."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        # Index first turn
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "first question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "first answer"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        count1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count1 >= 1

        old_ids = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
        old_texts = {r[0] for r in conn.execute("SELECT chunk_text FROM chunks").fetchall()}
        assert any("first" in t for t in old_texts)

        # Append second turn
        time.sleep(0.1)  # ensure mtime changes
        _append_lines(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "second question"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "second answer"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])

        index_file(conn, model, path, "proj")
        count2 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count2 >= 2, f"Expected at least 2 chunks, got {count2}"

        # Old chunks must still exist with same IDs
        new_ids = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
        assert old_ids.issubset(new_ids), "Old chunk IDs should be preserved"

        # Both turns must be present
        all_texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks ORDER BY message_index").fetchall()]
        assert any("first" in t for t in all_texts), "First turn should still exist"
        assert any("second" in t for t in all_texts), "Second turn should be added"

        # message_index should be distinct
        indices = [r[0] for r in conn.execute(
            "SELECT DISTINCT message_index FROM chunks ORDER BY message_index"
        ).fetchall()]
        assert len(indices) >= 2, f"Expected distinct message indices, got {indices}"
        assert indices[0] != indices[1], "Message indices must not collide"

        conn.close()

def test_dangling_user_not_lost():
    """If file ends with user message (no assistant yet), that turn is picked up on next run."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        # Write complete turn + dangling user
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "complete question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "complete answer"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "dangling question"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
        ])
        index_file(conn, model, path, "proj")
        count1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count1 >= 1  # only the complete turn

        # Now assistant arrives
        time.sleep(0.1)
        _append_lines(path, [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "dangling answer"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])

        index_file(conn, model, path, "proj")
        count2 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count2 >= 2, f"Expected at least 2 chunks after dangling resolved, got {count2}"

        all_texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks").fetchall()]
        assert any("dangling" in t for t in all_texts), "Dangling turn should now be indexed"

        conn.close()

def test_row_consistency_chunks_vec_fts():
    """chunks, chunks_vec, and chunks_fts row counts must match after indexing."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q1"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a1"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")

        # Append and reindex
        time.sleep(0.1)
        _append_lines(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q2"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a2"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])
        index_file(conn, model, path, "proj")

        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vec_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

        assert chunks_count == vec_count, f"chunks={chunks_count} != vec={vec_count}"
        assert chunks_count == fts_count, f"chunks={chunks_count} != fts={fts_count}"

        # No orphan rowids in vec
        orphan_vec = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT id FROM chunks)"
        ).fetchone()[0]
        assert orphan_vec == 0, f"Found {orphan_vec} orphan vec rows"

        conn.close()
