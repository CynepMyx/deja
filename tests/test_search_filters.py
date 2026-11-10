"""Filters must narrow the candidate set before ranking, not after.

Synthetic vectors and a fake model keep these tests deterministic: real
embeddings would need hundreds of chunks to push the target past top-k.
"""
import os

import numpy as np
import pytest

import deja.search as search_mod
from deja.db import init_db, serialize_f32
from deja.search import hybrid_search

DIM = 384


def _unit(*pairs):
    v = np.zeros(DIM, dtype=np.float32)
    for idx, val in pairs:
        v[idx] = val
    return v / np.linalg.norm(v)


QUERY_VEC = _unit((0, 1.0))


class FakeModel:
    def embed(self, texts):
        for _ in texts:
            yield QUERY_VEC


@pytest.fixture
def conn(tmp_path):
    c = init_db(os.path.join(tmp_path, "t.db"))
    yield c
    c.close()


def _insert(conn, n, ts, text, vec, project="p", source="claude-code", git_branch=None):
    cur = conn.execute(
        """INSERT INTO chunks
        (session_id, message_index, split_index, timestamp, project_path,
         chunk_text, tool_result_text, source, git_branch)
        VALUES (?, ?, 0, ?, ?, ?, '', ?, ?)""",
        ("s", n, ts, project, text, source, git_branch),
    )
    cid = cur.lastrowid
    conn.execute("INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                 (cid, serialize_f32(vec.tolist())))
    conn.execute("INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, '')",
                 (cid, text))
    return cid


def _crowd(conn, crowd=150, ts="2026-08-02T16:49:32.000Z", **target_kw):
    """`crowd` near-perfect matches on 2026-10-15, one weaker target (by default on 2026-08-02)."""
    for i in range(crowd):
        _insert(conn, i, "2026-10-15T10:00:00.000Z",
                f"monitor monitor monitor filler {i}", _unit((0, 1.0), (1 + i, 0.01)))
    target = _insert(conn, crowd, ts,
                     "target chunk, long text that mentions monitor once among other words",
                     _unit((0, 1.0), (300, 1.0)), **target_kw)
    conn.commit()
    return target


def _ids(results):
    return [r["id"] for r in results]


def test_narrow_date_finds_chunk_outside_global_top_k(conn):
    target = _crowd(conn)
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5,
                            date_from="2026-08-02", date_to="2026-08-03")
    assert _ids(results) == [target]


def test_date_only_date_to_includes_that_day(conn):
    target = _crowd(conn, crowd=3)
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5,
                            date_from="2026-08-02", date_to="2026-08-02")
    assert _ids(results) == [target]


def test_full_timestamp_date_to_stays_exact(conn):
    target = _crowd(conn, crowd=3)
    assert hybrid_search(conn, FakeModel(), "monitor", limit=5,
                         date_from="2026-08-02", date_to="2026-08-02T16:00:00Z") == []
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5,
                            date_from="2026-08-02", date_to="2026-08-02T17:00:00Z")
    assert _ids(results) == [target]


def test_project_filter_finds_small_project_outside_top_k(conn):
    target = _crowd(conn, project="small", ts="2026-10-15T11:00:00.000Z")
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5, project="small")
    assert _ids(results) == [target]


def test_filters_combine_project_and_date(conn):
    target = _crowd(conn)
    _insert(conn, 201, "2026-08-02T18:00:00.000Z", "monitor elsewhere", _unit((0, 1.0)),
            project="other")
    conn.commit()
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5, project="p",
                            date_from="2026-08-02", date_to="2026-08-02")
    assert _ids(results) == [target]


def test_git_branch_prefix_is_literal_not_like_pattern(conn):
    target = _crowd(conn, crowd=3, git_branch="fix_1/x")
    _insert(conn, 99, "2026-08-02T10:00:00.000Z", "monitor other",
            _unit((0, 1.0)), git_branch="fixA1/y")
    conn.commit()
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5, git_branch_prefix="fix_1/")
    assert _ids(results) == [target]


def test_wide_filter_uses_knn_fallback(conn, monkeypatch):
    """Above the candidate cap, KNN plus filter still returns the filtered set by distance."""
    monkeypatch.setattr(search_mod, "PREFILTER_MAX_CANDIDATES", 10)
    target = _crowd(conn, crowd=40)
    _insert(conn, 200, "2026-10-15T12:00:00.000Z", "other project", _unit((0, 1.0)),
            project="other")
    conn.commit()
    results = hybrid_search(conn, FakeModel(), "monitor", limit=5, project="p")
    assert len(results) == 5
    assert all(r["project_path"] == "p" for r in results)
    assert target not in _ids(results)


def test_fts_leg_applies_filters_before_limit(conn):
    target = _crowd(conn)
    where, params = search_mod._filter_clause(date_from="2026-08-02", date_to="2026-08-02")
    results = search_mod._fts_search(conn, "monitor", k=5, where=where, params=params)
    assert _ids(results) == [target]
