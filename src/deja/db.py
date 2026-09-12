import importlib.metadata
import os
import sqlite3
import struct
import sys

import sqlite_vec

SCHEMA_VERSION = 4
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Markers that describe what produced the rows in an index. Everything in
# REBUILD_KEYS invalidates existing chunks or embeddings when it changes, so a
# mismatch makes `deja index` rebuild rather than resume. `writer_version` is
# recorded but does not trigger a rebuild on its own: most releases change
# nothing about what is stored, and a two-hour reindex per patch release is a
# guard people switch off.
REBUILD_KEYS = ("schema_version", "parser_version", "embedding_model", "embedding_dim")


def writer_version() -> str:
    """Version of the deja build writing this index.

    Read from installed package metadata rather than a hand-kept constant:
    `deja/__init__.py` and `pyproject.toml` have already drifted apart once,
    and a provenance marker that can lie is worse than no marker at all.
    """
    try:
        return importlib.metadata.version("dejasearch")
    except importlib.metadata.PackageNotFoundError:
        return "0+source"


def index_identity() -> dict:
    """The markers this build would stamp into an index."""
    from deja.parsers.registry import PARSER_VERSION

    return {
        "schema_version": str(SCHEMA_VERSION),
        "parser_version": str(PARSER_VERSION),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": str(EMBEDDING_DIM),
        "writer_version": writer_version(),
    }


def read_identity(db_path: str) -> dict:
    """Markers stored in an index, read without creating or migrating it.

    `init_db` migrates and re-stamps, so anything that wants to compare the
    stored markers against this build has to read them first — otherwise the
    comparison can only ever agree with itself.
    """
    if not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        return {}
    try:
        return dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.OperationalError:
        # No meta table: zero-byte, half-copied, or an interrupted first run.
        return {}
    finally:
        conn.close()


def stale_identity(stored: dict) -> list[str]:
    """Rebuild-triggering markers on which a stored index disagrees with us."""
    current = index_identity()
    return [key for key in REBUILD_KEYS if stored.get(key) != current[key]]


def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)

def _migrate_if_needed(conn: sqlite3.Connection):
    """Drop index tables when the on-disk schema version differs.

    Tables are recreated by init_db right after; data is rebuilt by the
    next `deja index`. Must run before any DDL that references new columns.
    """
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if not has_meta:
        return

    db_version = int(get_meta(conn).get("schema_version", "0"))
    if db_version == SCHEMA_VERSION:
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
        CREATE INDEX IF NOT EXISTS idx_chunks_branch ON chunks(git_branch);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
    """)

    # Re-stamped on every open, not inserted once. With INSERT OR IGNORE the
    # marker described whichever build first created the file and nothing
    # after it, which is how an index came to claim a schema version it had
    # never had while its rows were written by an entirely different parser.
    for key, value in index_identity().items():
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
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
