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
