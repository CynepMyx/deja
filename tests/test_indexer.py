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
        conn.execute(
            "UPDATE indexed_files SET last_offset = 99999 WHERE path = ?",
            (path,),
        )
        conn.commit()
        needs = check_needs_reindex(conn, path)
        assert needs == "full"
        conn.close()
