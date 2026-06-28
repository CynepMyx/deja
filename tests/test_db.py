import os
import sqlite3
import tempfile

import sqlite_vec

from deja.db import init_db, get_meta, SCHEMA_VERSION

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
        assert meta["schema_version"] == str(SCHEMA_VERSION)
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

def _make_v1_db(db_path):
    """Build a DB exactly as schema v1 created it: chunks without `source`."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript("""
        CREATE TABLE chunks (
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
        CREATE TABLE indexed_files (
            path TEXT PRIMARY KEY,
            last_offset INTEGER NOT NULL DEFAULT 0,
            last_mtime REAL NOT NULL,
            last_size INTEGER NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX idx_chunks_session ON chunks(session_id);
        CREATE INDEX idx_chunks_project_time ON chunks(project_path, timestamp);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO chunks (session_id, message_index, split_index, chunk_text)
            VALUES ('old-sess', 0, 0, 'v1 data');
        INSERT INTO indexed_files VALUES ('/old/path.jsonl', 100, 1.0, 100);
    """)
    conn.execute("CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[384])")
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_text, tool_result_text, tokenize = "unicode61 tokenchars '-._/:'"
        )
    """)
    conn.commit()
    conn.close()

def test_init_db_migrates_v1_database():
    """Opening a v1 DB with current code must not crash and must rebuild schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        _make_v1_db(db_path)

        conn = init_db(db_path)

        meta = get_meta(conn)
        assert meta["schema_version"] == str(SCHEMA_VERSION)

        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        assert "source" in cols
        cols = [r[1] for r in conn.execute("PRAGMA table_info(indexed_files)").fetchall()]
        assert "source" in cols

        # Old data dropped — stale offsets must not survive into the new schema
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 0
        conn.close()

def test_init_db_idempotent_on_current_schema():
    """Re-opening a current-version DB must not drop data."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO chunks (session_id, message_index, split_index, chunk_text) "
            "VALUES ('s1', 0, 0, 'keep me')"
        )
        conn.commit()
        conn.close()

        conn = init_db(db_path)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        conn.close()
