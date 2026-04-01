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

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_project_time ON chunks(project_path, timestamp);
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
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn
