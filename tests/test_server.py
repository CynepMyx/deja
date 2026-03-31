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
