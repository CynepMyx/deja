"""Read-only analytics over the pre-aggregated sessions and tool_calls tables."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone


def top_sessions_by_cost(conn: sqlite3.Connection, n: int = 10) -> list[dict]:
    rows = conn.execute(
        """SELECT session_id, source, project_path, started_at, ended_at, turn_count,
                  input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                  (input_tokens + output_tokens + cache_creation_tokens) AS total_cost
           FROM sessions
           ORDER BY total_cost DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    return [
        {
            "session_id": r[0], "source": r[1], "project_path": r[2],
            "started_at": r[3], "ended_at": r[4], "turn_count": r[5],
            "input_tokens": r[6], "output_tokens": r[7],
            "cache_creation_tokens": r[8], "cache_read_tokens": r[9],
            "total_cost": r[10],
        }
        for r in rows
    ]


def top_sessions_by_length(conn: sqlite3.Connection, n: int = 10) -> list[dict]:
    rows = conn.execute(
        """SELECT session_id, source, project_path, started_at, ended_at, turn_count
           FROM sessions
           ORDER BY turn_count DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    return [
        {
            "session_id": r[0], "source": r[1], "project_path": r[2],
            "started_at": r[3], "ended_at": r[4], "turn_count": r[5],
        }
        for r in rows
    ]


def by_project(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT COALESCE(project_path, '(unknown)') AS project,
                  COUNT(*) AS sessions,
                  SUM(turn_count) AS turns,
                  SUM(input_tokens + output_tokens + cache_creation_tokens) AS total_cost
           FROM sessions
           GROUP BY project
           ORDER BY total_cost DESC"""
    ).fetchall()
    return [
        {"project": r[0], "sessions": r[1], "turns": r[2], "total_cost": r[3] or 0}
        for r in rows
    ]


def by_tool(conn: sqlite3.Connection, n: int = 10) -> list[dict]:
    rows = conn.execute(
        """SELECT tool_name, SUM(call_count) AS total_calls,
                  COUNT(DISTINCT session_id) AS in_sessions
           FROM tool_calls
           GROUP BY tool_name
           ORDER BY total_calls DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    return [
        {"tool_name": r[0], "total_calls": r[1], "in_sessions": r[2]}
        for r in rows
    ]


def by_day(conn: sqlite3.Connection, since_days: int = 30) -> list[dict]:
    """One row per day for the last N days. Counts sessions that started that day."""
    rows = conn.execute(
        """SELECT substr(started_at, 1, 10) AS day,
                  COUNT(*) AS sessions,
                  SUM(turn_count) AS turns,
                  SUM(input_tokens + output_tokens + cache_creation_tokens) AS total_cost
           FROM sessions
           WHERE started_at IS NOT NULL
             AND substr(started_at, 1, 10) >= ?
           GROUP BY day
           ORDER BY day"""
        ,
        ((datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d"),),
    ).fetchall()
    return [
        {"day": r[0], "sessions": r[1], "turns": r[2] or 0, "total_cost": r[3] or 0}
        for r in rows
    ]


def collect_all(conn: sqlite3.Connection, top: int = 10, since_days: int = 30) -> dict:
    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_turns = conn.execute("SELECT COALESCE(SUM(turn_count), 0) FROM sessions").fetchone()[0]
    total_cost = conn.execute(
        "SELECT COALESCE(SUM(input_tokens + output_tokens + cache_creation_tokens), 0) FROM sessions"
    ).fetchone()[0]
    return {
        "totals": {
            "sessions": total_sessions,
            "turns": total_turns,
            "total_cost_tokens": total_cost,
        },
        "top_cost": top_sessions_by_cost(conn, top),
        "top_length": top_sessions_by_length(conn, top),
        "by_project": by_project(conn),
        "by_tool": by_tool(conn, top),
        "by_day": by_day(conn, since_days),
    }


# ----- human formatting -----

def _sparkline(counts: list[int]) -> str:
    if not counts:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    hi = max(counts)
    if hi == 0:
        return bars[0] * len(counts)
    return "".join(bars[min(len(bars) - 1, int(c * (len(bars) - 1) / hi))] for c in counts)


def _fmt_num(n) -> str:
    return f"{int(n or 0):,}".replace(",", " ")


def format_human(report: dict) -> str:
    out: list[str] = []
    t = report["totals"]
    out.append(f"Sessions:     {_fmt_num(t['sessions'])}")
    out.append(f"Turns:        {_fmt_num(t['turns'])}")
    out.append(f"Total tokens: {_fmt_num(t['total_cost_tokens'])} (input + output + cache_creation)")
    out.append("")

    out.append("--- Top by cost ---")
    for r in report["top_cost"]:
        sid = (r["session_id"] or "?")[:12]
        ts = (r["started_at"] or "")[:10]
        proj = (r["project_path"] or "(unknown)")[:30]
        out.append(f"  {_fmt_num(r['total_cost']):>14}  {sid}  {ts}  {r['turn_count']:>4} turns  {proj}")
    out.append("")

    out.append("--- Top by length ---")
    for r in report["top_length"]:
        sid = (r["session_id"] or "?")[:12]
        ts = (r["started_at"] or "")[:10]
        proj = (r["project_path"] or "(unknown)")[:30]
        out.append(f"  {r['turn_count']:>6} turns  {sid}  {ts}  {proj}")
    out.append("")

    out.append("--- By project ---")
    for r in report["by_project"]:
        proj = (r["project"] or "(unknown)")[:50]
        out.append(f"  {_fmt_num(r['total_cost']):>14}  sessions={r['sessions']:>4}  turns={_fmt_num(r['turns']):>10}  {proj}")
    out.append("")

    out.append("--- Tool usage (top) ---")
    for r in report["by_tool"]:
        out.append(f"  {r['total_calls']:>6} calls  in {r['in_sessions']:>3} sessions  {r['tool_name']}")
    out.append("")

    by_day_data = report["by_day"]
    if by_day_data:
        out.append("--- Sessions per day ---")
        counts = [d["sessions"] for d in by_day_data]
        out.append(f"  {by_day_data[0]['day']} .. {by_day_data[-1]['day']}  {_sparkline(counts)}  (max {max(counts)})")
    return "\n".join(out)


def format_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
