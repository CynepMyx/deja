import json
import os
import tempfile
from deja.db import init_db
from deja.indexer import get_embedding_model, index_file
from deja.server import _do_search, _do_get_session, _do_get_context

def _make_test_index(tmp, multi_turn=False):
    db_path = os.path.join(tmp, "test.db")
    conn = init_db(db_path)
    model = get_embedding_model()
    path = os.path.join(tmp, "sess-001.jsonl")
    lines = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Fix SSL certificate"}]}, "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Run certbot renew"}]}, "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
    ]
    if multi_turn:
        lines.extend([
            {"type": "user", "message": {"content": [{"type": "text", "text": "Check nginx config"}]}, "timestamp": "2026-03-30T10:01:00Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "nginx -t shows ok"}]}, "timestamp": "2026-03-30T10:01:05Z", "uuid": "4"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "Deploy to production"}]}, "timestamp": "2026-03-30T10:02:00Z", "uuid": "5"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Deployed successfully"}]}, "timestamp": "2026-03-30T10:02:05Z", "uuid": "6"},
        ])
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


def test_get_context_returns_window():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_test_index(tmp, multi_turn=True)
        # Get the middle chunk (message_index=1)
        row = conn.execute(
            "SELECT id FROM chunks WHERE session_id = 'sess-001' AND message_index = 1"
        ).fetchone()
        assert row is not None
        chunk_id = row[0]

        anchor_id, chunks = _do_get_context(conn, chunk_id, window=1)
        assert anchor_id == chunk_id
        msg_indices = {c["message_index"] for c in chunks}
        assert 0 in msg_indices  # prev turn
        assert 1 in msg_indices  # anchor
        assert 2 in msg_indices  # next turn

        anchor_chunks = [c for c in chunks if c["is_anchor"]]
        assert len(anchor_chunks) == 1
        conn.close()


def test_get_context_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_test_index(tmp)
        anchor_id, chunks = _do_get_context(conn, 999999, window=2)
        assert anchor_id is None
        assert chunks == []
        conn.close()
