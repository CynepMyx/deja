import json
import os
import tempfile

from deja.analytics import (
    collect_all, top_sessions_by_cost, top_sessions_by_length,
    by_project, by_tool, by_day, format_human, format_json,
)
from deja.db import init_db


def _seed_session(conn, **kwargs):
    defaults = dict(
        session_id="s1", source="claude-code", project_path="proj-a",
        started_at="2026-06-01T10:00:00Z", ended_at="2026-06-01T11:00:00Z",
        turn_count=5, input_tokens=100, output_tokens=200,
        cache_creation_tokens=50, cache_read_tokens=10,
    )
    defaults.update(kwargs)
    conn.execute(
        """INSERT INTO sessions (session_id, source, project_path, started_at, ended_at,
                                 turn_count, input_tokens, output_tokens,
                                 cache_creation_tokens, cache_read_tokens)
           VALUES (:session_id, :source, :project_path, :started_at, :ended_at,
                   :turn_count, :input_tokens, :output_tokens,
                   :cache_creation_tokens, :cache_read_tokens)""",
        defaults,
    )


def _seed_tool(conn, session_id, tool_name, count):
    conn.execute(
        "INSERT INTO tool_calls (session_id, tool_name, call_count) VALUES (?, ?, ?)",
        (session_id, tool_name, count),
    )


def test_top_cost_orders_by_input_output_cache_creation():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="s_low", input_tokens=10, output_tokens=10, cache_creation_tokens=0)
        _seed_session(conn, session_id="s_high", input_tokens=1000, output_tokens=2000, cache_creation_tokens=500)
        _seed_session(conn, session_id="s_mid", input_tokens=500, output_tokens=500, cache_creation_tokens=100)
        conn.commit()

        rows = top_sessions_by_cost(conn, n=10)
        assert [r["session_id"] for r in rows] == ["s_high", "s_mid", "s_low"]
        assert rows[0]["total_cost"] == 1000 + 2000 + 500
        conn.close()


def test_top_cost_excludes_cache_read():
    """cache_read_tokens не входит в total_cost (это re-use, не новые токены)."""
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="s", input_tokens=0, output_tokens=0,
                      cache_creation_tokens=0, cache_read_tokens=999999)
        conn.commit()
        rows = top_sessions_by_cost(conn, n=10)
        assert rows[0]["total_cost"] == 0
        conn.close()


def test_top_length_orders_by_turn_count():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="short", turn_count=5)
        _seed_session(conn, session_id="long", turn_count=500)
        _seed_session(conn, session_id="mid", turn_count=50)
        conn.commit()
        rows = top_sessions_by_length(conn, n=10)
        assert [r["session_id"] for r in rows] == ["long", "mid", "short"]
        conn.close()


def test_by_project_aggregates():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="a1", project_path="proj-a", turn_count=10, input_tokens=100, output_tokens=100, cache_creation_tokens=0)
        _seed_session(conn, session_id="a2", project_path="proj-a", turn_count=20, input_tokens=200, output_tokens=200, cache_creation_tokens=0)
        _seed_session(conn, session_id="b1", project_path="proj-b", turn_count=5, input_tokens=50, output_tokens=50, cache_creation_tokens=0)
        conn.commit()

        rows = by_project(conn)
        rows_by_proj = {r["project"]: r for r in rows}
        assert rows_by_proj["proj-a"]["sessions"] == 2
        assert rows_by_proj["proj-a"]["turns"] == 30
        assert rows_by_proj["proj-a"]["total_cost"] == 600
        assert rows_by_proj["proj-b"]["sessions"] == 1
        conn.close()


def test_by_project_handles_null_path():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="x", project_path=None)
        conn.commit()
        rows = by_project(conn)
        assert rows[0]["project"] == "(unknown)"
        conn.close()


def test_by_tool_orders_by_total_calls():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_tool(conn, "s1", "Bash", 100)
        _seed_tool(conn, "s2", "Bash", 50)
        _seed_tool(conn, "s1", "Read", 200)
        _seed_tool(conn, "s1", "Edit", 10)
        conn.commit()
        rows = by_tool(conn, n=10)
        assert rows[0]["tool_name"] == "Read"
        assert rows[0]["total_calls"] == 200
        assert rows[0]["in_sessions"] == 1
        assert rows[1]["tool_name"] == "Bash"
        assert rows[1]["total_calls"] == 150
        assert rows[1]["in_sessions"] == 2
        conn.close()


def test_by_day_filters_window():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="old", started_at="2020-01-01T00:00:00Z")
        _seed_session(conn, session_id="recent", started_at="2099-12-31T23:00:00Z")
        conn.commit()
        rows = by_day(conn, since_days=30)
        days = [r["day"] for r in rows]
        assert "2020-01-01" not in days
        # recent should be in (future date never gets filtered out by since cutoff)
        assert "2099-12-31" in days
        conn.close()


def test_collect_all_includes_totals():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn, session_id="a", turn_count=10, input_tokens=100, output_tokens=200, cache_creation_tokens=50)
        _seed_session(conn, session_id="b", turn_count=20, input_tokens=200, output_tokens=400, cache_creation_tokens=100)
        conn.commit()
        report = collect_all(conn)
        assert report["totals"]["sessions"] == 2
        assert report["totals"]["turns"] == 30
        assert report["totals"]["total_cost_tokens"] == (100+200+50) + (200+400+100)
        conn.close()


def test_format_human_runs():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn)
        _seed_tool(conn, "s1", "Bash", 5)
        conn.commit()
        out = format_human(collect_all(conn))
        assert "Sessions:" in out
        assert "Top by cost" in out
        assert "By project" in out
        assert "Tool usage" in out
        conn.close()


def test_format_json_valid():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        _seed_session(conn)
        conn.commit()
        out = format_json(collect_all(conn))
        data = json.loads(out)
        assert "totals" in data
        assert "top_cost" in data
        assert data["totals"]["sessions"] == 1
        conn.close()


# Integration test (full indexer → sessions/tool_calls/git_branch) deferred —
# requires fastembed model (117MB) + suffers from Windows SQLite WAL file lock
# on tempdir cleanup. Covered by end-to-end reindex in Task D before release.
