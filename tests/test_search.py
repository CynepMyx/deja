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
    assert fts5_escape('test "quotes"') == '"test" AND """quotes"""'

def test_fts5_escape_token_and():
    assert fts5_escape('nginx proxy') == '"nginx" AND "proxy"'
    assert fts5_escape('single') == '"single"'
    assert fts5_escape('') == '""'
