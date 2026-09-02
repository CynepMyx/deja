import sqlite3
import struct
import sys

import sqlite_vec

SCHEMA_VERSION = 5
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)

# Version -> statements that upgrade the previous version in place. A version
# listed here adds columns only, so the existing rows and — crucially — the
# existing embeddings survive. Anything not listed falls back to a rebuild.
ADDITIVE_MIGRATIONS = {
    # Sub-agent threads were not indexed before v5, so every existing row is
    # correctly a 'main' row and the column default needs no backfill.
    5: [
        "ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'main'",
        "ALTER TABLE sessions ADD COLUMN kind TEXT NOT NULL DEFAULT 'main'",
        "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT",
    ],
}


def _try_additive_migration(conn: sqlite3.Connection, db_version: int) -> bool:
    """Upgrade in place if every step from db_version to HEAD is additive.

    Returns False without touching the database if any step is missing or
    fails, leaving the caller to fall back to dropping and rebuilding.
    """
    steps = []
    for version in range(db_version + 1, SCHEMA_VERSION + 1):
        if version not in ADDITIVE_MIGRATIONS:
            return False
        steps.extend(ADDITIVE_MIGRATIONS[version])

    try:
        conn.execute("SAVEPOINT additive_migration")
        for statement in steps:
            conn.execute(statement)
        conn.execute("RELEASE additive_migration")
    except sqlite3.Error as e:
        conn.execute("ROLLBACK TO additive_migration")
        conn.execute("RELEASE additive_migration")
        print(f"[deja] in-place migration failed ({e}), rebuilding", file=sys.stderr)
        return False

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    print(
        f"[deja] schema v{db_version} -> v{SCHEMA_VERSION}: "
        "migrated in place, embeddings kept",
        file=sys.stderr,
    )
    return True


def _migrate_if_needed(conn: sqlite3.Connection):
    """Bring the on-disk schema to SCHEMA_VERSION.

    Additive upgrades are applied in place. Anything else drops the index
    tables — recreated by init_db right after, repopulated by the next
    `deja index`. Must run before any DDL that references new columns.
    """
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if not has_meta:
        return

    db_version = int(get_meta(conn).get("schema_version", "0"))
    if db_version == SCHEMA_VERSION:
        return

    if _try_additive_migration(conn, db_version):
        return

    print(
        f"[deja] schema v{db_version} -> v{SCHEMA_VERSION}: "
        "rebuilding index tables, run 'deja index' to repopulate",
        file=sys.stderr,
    )
    conn.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS indexed_files;
        DROP TABLE IF EXISTS chunks_vec;
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS sessions;
        DROP TABLE IF EXISTS tool_calls;
    """)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    _migrate_if_needed(conn)

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
            source TEXT NOT NULL DEFAULT 'claude-code',
            kind TEXT NOT NULL DEFAULT 'main',
            git_branch TEXT,
            parent_id INTEGER REFERENCES chunks(id),
            UNIQUE(session_id, message_index, split_index)
        );

        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            last_offset INTEGER NOT NULL DEFAULT 0,
            last_mtime REAL NOT NULL,
            last_size INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'claude-code',
            next_message_index INTEGER
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'main',
            parent_session_id TEXT,
            project_path TEXT,
            started_at TEXT,
            ended_at TEXT,
            turn_count INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            call_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (session_id, tool_name)
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
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
        CREATE INDEX IF NOT EXISTS idx_chunks_branch ON chunks(git_branch);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
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
