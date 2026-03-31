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
