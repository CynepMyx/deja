import json
import os
import tempfile

from deja.db import init_db
from deja.chunker import make_chunks
from deja.indexer import get_embedding_model, index_file
from deja.parsers import claude_code
from deja.search import hybrid_search


def _write_session(path: str, question: str, answer: str):
    # Two turns: the parser only finalizes a turn once the next one starts,
    # so a single-turn file would stay provisional and never reach `sessions`.
    lines = [
        {"type": "user", "message": {"content": [{"type": "text", "text": question}]},
         "timestamp": "2026-03-30T10:00:00Z", "uuid": "1"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]},
         "timestamp": "2026-03-30T10:00:05Z", "uuid": "2"},
        {"type": "user", "message": {"content": [{"type": "text", "text": "and then?"}]},
         "timestamp": "2026-03-30T10:01:00Z", "uuid": "3"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]},
         "timestamp": "2026-03-30T10:01:05Z", "uuid": "4"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def test_discover_yields_main_and_subagent_threads(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        project = os.path.join(tmp, "-home-user-proj")
        subagents = os.path.join(project, "session-1", "subagents")
        os.makedirs(subagents)
        main = os.path.join(project, "session-1.jsonl")
        sub = os.path.join(subagents, "agent-research-abc123.jsonl")
        _write_session(main, "q", "a")
        _write_session(sub, "q", "a")

        monkeypatch.setattr(claude_code, "CLAUDE_PROJECTS_DIR", tmp)
        found = {os.path.basename(p): k for p, _, k in claude_code.discover()}

        assert found == {
            "session-1.jsonl": "main",
            "agent-research-abc123.jsonl": "subagent",
        }
        assert {proj for _, proj, _ in claude_code.discover()} == {"-home-user-proj"}


def test_discover_ignores_unrelated_nested_jsonl(monkeypatch):
    """Only `<session>/subagents/*.jsonl` counts — not any nested file."""
    with tempfile.TemporaryDirectory() as tmp:
        project = os.path.join(tmp, "-home-user-proj")
        os.makedirs(os.path.join(project, "session-1", "backups"))
        stray = os.path.join(project, "session-1", "backups", "old.jsonl")
        _write_session(stray, "q", "a")

        monkeypatch.setattr(claude_code, "CLAUDE_PROJECTS_DIR", tmp)
        assert list(claude_code.discover()) == []


def test_chunks_carry_kind():
    turn = {"user_text": "q", "assistant_text": "a", "message_index": 0,
            "timestamp": "2026-03-30T10:00:00Z", "tool_result_text": ""}
    chunks = make_chunks(turn, "s1", "proj", kind="subagent")
    assert all(c["kind"] == "subagent" for c in chunks)
    assert make_chunks(turn, "s1", "proj")[0]["kind"] == "main"


def test_kind_defaults_to_main_in_schema():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        conn.execute(
            "INSERT INTO chunks (session_id, message_index, split_index, chunk_text)"
            " VALUES ('s', 0, 0, 'text')"
        )
        conn.execute(
            "INSERT INTO sessions (session_id, source) VALUES ('s', 'claude-code')"
        )
        assert conn.execute("SELECT kind FROM chunks").fetchone()[0] == "main"
        assert conn.execute("SELECT kind FROM sessions").fetchone()[0] == "main"
        conn.close()


def _indexed_db(tmp):
    conn = init_db(os.path.join(tmp, "t.db"))
    model = get_embedding_model()
    main = os.path.join(tmp, "main.jsonl")
    sub = os.path.join(tmp, "agent-sub.jsonl")
    _write_session(main, "How do I configure nginx?", "Edit nginx.conf and reload")
    _write_session(sub, "How do I configure nginx?", "Set proxy_pass in nginx.conf")
    index_file(conn, model, main, "proj", kind="main")
    index_file(conn, model, sub, "proj", kind="subagent")
    return conn, model


def test_search_excludes_subagents_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _indexed_db(tmp)
        results = hybrid_search(conn, model, "nginx", limit=10)
        assert results
        assert {r["kind"] for r in results} == {"main"}
        conn.close()


def test_search_includes_subagents_on_demand():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _indexed_db(tmp)
        results = hybrid_search(conn, model, "nginx", limit=10, include_subagents=True)
        assert {r["kind"] for r in results} == {"main", "subagent"}
        conn.close()


def test_indexed_subagent_rows_are_tagged():
    with tempfile.TemporaryDirectory() as tmp:
        conn, _ = _indexed_db(tmp)
        by_kind = dict(
            conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()
        )
        assert by_kind["main"] > 0 and by_kind["subagent"] > 0
        assert conn.execute(
            "SELECT kind FROM sessions WHERE session_id = 'agent-sub'"
        ).fetchone()[0] == "subagent"
        conn.close()


def test_subagent_exclusion_is_pushed_into_both_lanes():
    """A post-filter would let sub-agent chunks eat every candidate slot."""
    from deja.search import _fts_search, _vector_search

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        model = get_embedding_model()
        # One main session against many sub-agent threads on the same topic.
        main = os.path.join(tmp, "main.jsonl")
        _write_session(main, "How do I configure nginx?", "Edit nginx.conf")
        index_file(conn, model, main, "proj", kind="main")
        for i in range(12):
            sub = os.path.join(tmp, f"agent-{i}.jsonl")
            _write_session(sub, "How do I configure nginx?", "nginx proxy_pass here")
            index_file(conn, model, sub, "proj", kind="subagent")

        # k smaller than the sub-agent population: a post-filter would return 0.
        assert _fts_search(conn, "nginx", k=3, exclude_kind="subagent")
        assert _vector_search(conn, model, "nginx", k=3, exclude_kind="subagent")
        assert hybrid_search(conn, model, "nginx", limit=5)
        conn.close()


def test_v4_index_migrates_in_place_without_dropping_embeddings():
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "t.db")
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO chunks (session_id, message_index, split_index, chunk_text)"
            " VALUES ('s', 0, 0, 'text')"
        )
        conn.execute("INSERT INTO chunks_vec (rowid, embedding) VALUES (1, ?)",
                     (b"\x00" * (384 * 4),))
        # Roll back to the pre-kind schema.
        conn.execute("DROP INDEX idx_chunks_kind")
        conn.execute("ALTER TABLE chunks DROP COLUMN kind")
        conn.execute("ALTER TABLE sessions DROP COLUMN kind")
        conn.execute("DROP INDEX idx_sessions_parent")
        conn.execute("ALTER TABLE sessions DROP COLUMN parent_session_id")
        conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()

        conn = init_db(db_path)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 1
        assert conn.execute("SELECT kind FROM chunks").fetchone()[0] == "main"
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "5"
        conn.close()


def test_analytics_excludes_subagent_sessions_by_default():
    from deja import analytics

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        for sid, kind, turns in (("main-1", "main", 5), ("agent-1", "subagent", 90)):
            conn.execute(
                "INSERT INTO sessions (session_id, source, kind, project_path,"
                " started_at, turn_count, input_tokens) VALUES (?, 'claude-code', ?,"
                " 'proj', '2026-03-30T10:00:00Z', ?, ?)",
                (sid, kind, turns, turns * 10),
            )
        conn.commit()

        default = analytics.collect_all(conn)
        assert default["totals"]["sessions"] == 1
        assert default["top_length"][0]["session_id"] == "main-1"

        full = analytics.collect_all(conn, include_subagents=True)
        assert full["totals"]["sessions"] == 2
        assert full["top_length"][0]["session_id"] == "agent-1"
        conn.close()


def test_parent_session_id_derived_from_path():
    assert claude_code.parent_session_id(
        os.path.join("p", "session-1", "subagents", "agent-x.jsonl")
    ) == "session-1"
    assert claude_code.parent_session_id(
        os.path.join("p", "session-1.jsonl")
    ) is None


def test_indexed_subagent_links_back_to_parent():
    with tempfile.TemporaryDirectory() as tmp:
        project = os.path.join(tmp, "-home-user-proj")
        subagents = os.path.join(project, "session-1", "subagents")
        os.makedirs(subagents)
        main = os.path.join(project, "session-1.jsonl")
        sub = os.path.join(subagents, "agent-research.jsonl")
        _write_session(main, "How do I configure nginx?", "Edit nginx.conf")
        _write_session(sub, "How do I configure nginx?", "Set proxy_pass")

        conn = init_db(os.path.join(tmp, "t.db"))
        model = get_embedding_model()
        index_file(conn, model, main, "proj", kind="main")
        index_file(conn, model, sub, "proj", kind="subagent")

        assert conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = 'agent-research'"
        ).fetchone()[0] == "session-1"
        assert conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = 'session-1'"
        ).fetchone()[0] is None

        hits = hybrid_search(conn, model, "nginx", limit=10, include_subagents=True)
        by_kind = {h["kind"]: h for h in hits}
        assert by_kind["subagent"]["parent_session_id"] == "session-1"
        assert by_kind["main"]["parent_session_id"] is None
        conn.close()


def test_list_subagent_threads_walks_the_link_back():
    from deja.server import _do_list_subagents

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        conn.execute(
            "INSERT INTO sessions (session_id, source, kind, parent_session_id,"
            " started_at) VALUES ('agent-1', 'claude-code', 'subagent', 'session-1',"
            " '2026-03-30T10:00:00Z')"
        )
        conn.execute(
            "INSERT INTO sessions (session_id, source, kind) "
            "VALUES ('session-1', 'claude-code', 'main')"
        )
        conn.commit()
        threads = _do_list_subagents(conn, "session-1")
        assert [t["session_id"] for t in threads] == ["agent-1"]
        assert _do_list_subagents(conn, "agent-1") == []
        conn.close()


def test_newer_schema_is_not_stamped_backwards():
    """A db from a future deja must not be relabelled as ours."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "t.db")
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO chunks (session_id, message_index, split_index, chunk_text)"
            " VALUES ('s', 0, 0, 'text')"
        )
        conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()

        conn = init_db(db_path)
        # Falls back to the rebuild path rather than silently claiming v5.
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        conn.close()


def test_vector_search_stops_when_the_index_is_exhausted():
    with tempfile.TemporaryDirectory() as tmp:
        from deja.search import _vector_search

        conn = init_db(os.path.join(tmp, "t.db"))
        model = get_embedding_model()
        sub = os.path.join(tmp, "agent-only.jsonl")
        _write_session(sub, "How do I configure nginx?", "proxy_pass")
        index_file(conn, model, sub, "proj", kind="subagent")

        calls = []

        class _Counting:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args):
                calls.append(sql)
                return self._inner.execute(sql, *args)

        counting = _Counting(conn)
        assert _vector_search(counting, model, "nginx", k=100, exclude_kind="subagent") == []
        knn_calls = [c for c in calls if "chunks_vec" in c]
        assert len(knn_calls) == 1, f"widened past an exhausted index: {len(knn_calls)}"
        conn.close()


def test_schema_guard_survives_an_index_without_a_meta_table():
    import sqlite3
    import pytest
    from deja.cli import _require_current_schema

    with tempfile.TemporaryDirectory() as tmp:
        conn = sqlite3.connect(os.path.join(tmp, "empty.db"))
        with pytest.raises(SystemExit):
            _require_current_schema(conn)
        conn.close()


def test_stale_index_is_described_not_raised_at_startup():
    """A schema mismatch must not kill the server: clients only see a closed pipe."""
    from deja.server import _schema_problem

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        assert _schema_problem(conn) is None
        conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
        problem = _schema_problem(conn)
        assert problem and "v4" in problem and "deja index" in problem
        conn.close()


def test_gc_removes_a_deleted_subagent_thread():
    from deja.indexer import gc_orphans

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        model = get_embedding_model()
        main = os.path.join(tmp, "main.jsonl")
        sub = os.path.join(tmp, "agent-gone.jsonl")
        _write_session(main, "How do I configure nginx?", "Edit nginx.conf")
        _write_session(sub, "How do I configure nginx?", "Set proxy_pass")
        index_file(conn, model, main, "proj", kind="main")
        index_file(conn, model, sub, "proj", kind="subagent")

        gc_orphans(conn, {main}, sources=["claude-code"])

        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE kind = 'subagent'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE kind = 'main'"
        ).fetchone()[0] > 0
        conn.close()


def test_server_startup_survives_an_index_without_a_meta_table():
    """The lifespan must not raise: an MCP client only sees a closed pipe."""
    import sqlite3
    from deja.server import _schema_problem

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.db")
        open(path, "wb").close()
        conn = sqlite3.connect(path)
        problem = _schema_problem(conn)
        assert problem and "v0" in problem and "deja index" in problem
        conn.close()
