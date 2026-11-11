"""A run that dies mid-file must not leave the file marked as fully indexed."""
import json
import os

import numpy as np
import pytest

import deja.indexer as indexer
from deja.db import init_db
from deja.indexer import check_needs_reindex, index_file


def _turn(n):
    return [
        {"type": "user", "message": {"content": [{"type": "text", "text": f"question {n}"}]},
         "timestamp": f"2026-01-01T00:{n:02d}:00Z", "uuid": f"u{n}"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"answer {n}"}]},
         "timestamp": f"2026-01-01T00:{n:02d}:01Z", "uuid": f"a{n}"},
    ]


def _write(path, turns):
    with open(path, "w", encoding="utf-8") as f:
        for n in range(turns):
            for rec in _turn(n):
                f.write(json.dumps(rec) + "\n")


class FakeModel:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed(self, texts):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("killed mid-run")
        for _ in texts:
            yield np.ones(384, dtype=np.float32)


@pytest.fixture
def conn(tmp_path):
    c = init_db(str(tmp_path / "t.db"))
    yield c
    c.close()


def _indexes(conn):
    return [r[0] for r in conn.execute("SELECT message_index FROM chunks ORDER BY message_index")]


def test_run_killed_after_first_batch_resumes_next_time(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "TURNS_PER_BATCH", 2)
    path = str(tmp_path / "s.jsonl")
    _write(path, 6)

    with pytest.raises(RuntimeError):
        index_file(conn, FakeModel(fail_on_call=2), path, "p")
    assert _indexes(conn) == [0, 1]

    assert check_needs_reindex(conn, path) == "incremental"
    index_file(conn, FakeModel(), path, "p")
    assert _indexes(conn) == [0, 1, 2, 3, 4, 5]


def test_completed_run_is_not_redone(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "TURNS_PER_BATCH", 2)
    path = str(tmp_path / "s.jsonl")
    _write(path, 5)
    index_file(conn, FakeModel(), path, "p")
    assert _indexes(conn) == [0, 1, 2, 3, 4]
    assert check_needs_reindex(conn, path) is False


def test_heal_marks_files_left_by_an_old_killed_run(conn, tmp_path, monkeypatch):
    """Rows written by the old code look complete (last_size == file size)."""
    from deja.indexer import mark_unfinished_files

    monkeypatch.setattr(indexer, "TURNS_PER_BATCH", 2)
    path = str(tmp_path / "s.jsonl")
    _write(path, 6)
    with pytest.raises(RuntimeError):
        index_file(conn, FakeModel(fail_on_call=2), path, "p")
    st = os.stat(path)
    conn.execute("UPDATE indexed_files SET last_size = ?, last_mtime = ? WHERE path = ?",
                 (st.st_size, st.st_mtime, path))
    conn.commit()
    assert check_needs_reindex(conn, path) is False

    done = str(tmp_path / "done.jsonl")
    _write(done, 3)
    index_file(conn, FakeModel(), done, "p")

    assert mark_unfinished_files(conn) == [path]
    index_file(conn, FakeModel(), path, "p")
    index_file(conn, FakeModel(), done, "p")
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's'").fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 'done'").fetchone()[0] == 3
