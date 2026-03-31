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
        assert meta["schema_version"] == "1"
        assert meta["embedding_model"] == "intfloat/multilingual-e5-small"
        assert meta["embedding_dim"] == "384"
        conn.close()

def test_wal_mode_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()
