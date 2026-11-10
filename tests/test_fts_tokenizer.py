"""File names and path fragments must be findable inside longer paths."""
import os

import pytest

from deja.cli import cmd_index
from deja.db import fts_needs_rebuild, init_db, rebuild_fts
from deja.search import _fts_search

PATH_TEXT = "cp /opt/registry-infra/scripts/monitor.sh /opt/registry-infra/scripts/monitor.sh.bak-20260802-164932"
OLD_FTS = """CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_text, tool_result_text, tokenize = "unicode61 tokenchars '-._/:'")"""


@pytest.fixture
def conn(tmp_path):
    c = init_db(os.path.join(tmp_path, "t.db"))
    yield c
    c.close()


def _add_chunk(conn, n, text, tool_text=""):
    cid = conn.execute(
        """INSERT INTO chunks (session_id, message_index, split_index, timestamp,
           project_path, chunk_text, tool_result_text)
           VALUES ('s', ?, 0, '2026-08-02T16:49:32Z', 'p', ?, ?)""",
        (n, text, tool_text),
    ).lastrowid
    conn.execute("INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                 (cid, text, tool_text or ""))
    conn.commit()
    return cid


def _found(conn, query):
    return [r["id"] for r in _fts_search(conn, query, k=10)]


@pytest.mark.parametrize("query", [
    "monitor.sh.bak-20260802-164932",
    "scripts/monitor.sh",
    "registry-infra",
    "/opt/registry-infra/scripts/monitor.sh.bak-20260802-164932",
])
def test_file_name_and_path_fragments_match(conn, query):
    cid = _add_chunk(conn, 0, PATH_TEXT)
    assert cid in _found(conn, query)


def test_new_index_uses_current_tokenizer(conn):
    assert not fts_needs_rebuild(conn)


def test_rebuild_migrates_old_tokenizer_without_touching_vectors(conn):
    conn.execute("DROP TABLE chunks_fts")
    conn.execute(OLD_FTS)
    cid = _add_chunk(conn, 0, PATH_TEXT)
    _add_chunk(conn, 1, "second", tool_text=None)
    conn.commit()
    vec_before = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]

    assert fts_needs_rebuild(conn)
    assert cid not in _found(conn, "monitor.sh.bak-20260802-164932")

    assert rebuild_fts(conn) == 2
    assert not fts_needs_rebuild(conn)
    assert cid in _found(conn, "monitor.sh.bak-20260802-164932")
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == vec_before


def test_rebuild_failure_keeps_old_table(conn, monkeypatch):
    conn.execute("DROP TABLE chunks_fts")
    conn.execute(OLD_FTS)
    _add_chunk(conn, 0, PATH_TEXT)

    import deja.db as db_mod

    def broken_create(c):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(db_mod, "_create_fts", broken_create)
    with pytest.raises(RuntimeError):
        rebuild_fts(conn)
    assert fts_needs_rebuild(conn)
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1


def test_cmd_index_rebuilds_old_fts(tmp_path, monkeypatch):
    import argparse

    import deja.cli as cli

    db_path = os.path.join(tmp_path, "index.db")
    c = init_db(db_path)
    c.execute("DROP TABLE chunks_fts")
    c.execute(OLD_FTS)
    _add_chunk(c, 0, PATH_TEXT)
    c.close()

    monkeypatch.setattr(cli, "get_index_dir", lambda: str(tmp_path))
    monkeypatch.setattr(cli, "get_index_path", lambda: db_path)
    monkeypatch.setattr(cli, "_acquire_lock", lambda: None)
    monkeypatch.setattr(cli, "_release_lock", lambda fd: None)
    monkeypatch.setattr(cli, "get_embedding_model", lambda: None)
    monkeypatch.setattr(cli, "_collect_files", lambda sources: [])
    monkeypatch.setattr(cli, "gc_orphans", lambda *a, **k: None)
    cmd_index(argparse.Namespace(reindex=False, source="all"))

    c = init_db(db_path)
    try:
        assert not fts_needs_rebuild(c)
        assert c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        c.close()
