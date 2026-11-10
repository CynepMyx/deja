"""A tool result belongs to the turn whose assistant step called the tool.

Claude Code writes tool results as `user` records after the call, so a turn
boundary sits between a call and its output.
"""
import json
import os

import numpy as np
import pytest

from deja.db import init_db
from deja.indexer import index_file
from deja.parser import parse_jsonl_file


def _u_text(text, n):
    return {"type": "user", "message": {"content": [{"type": "text", "text": text}]},
            "timestamp": f"2026-01-01T00:00:{n:02d}Z", "uuid": f"u{n}"}


def _u_result(tool_id, out, n):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": out}]},
        "timestamp": f"2026-01-01T00:00:{n:02d}Z", "uuid": f"r{n}"}


def _a_tool(tool_id, cmd, n):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": cmd}}]},
        "timestamp": f"2026-01-01T00:00:{n:02d}Z", "uuid": f"a{n}"}


def _a_text(text, n):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]},
            "timestamp": f"2026-01-01T00:00:{n:02d}Z", "uuid": f"t{n}"}


SESSION = [
    _u_text("check the backup", 0),
    _a_tool("A", "ls /backup", 1),
    _u_result("A", "OUT_A db.sql.gz", 2),
    _a_tool("B", "cat monitor.sh", 3),
    _a_tool("C", "df -h", 4),
    _u_result("B", "OUT_B monitor body", 5),
    _u_result("C", "OUT_C disk 42%", 6),
    _a_text("Backup is fine.", 7),
    _u_text("thanks, now the logs", 8),
    _a_tool("D", "tail log", 9),
    _u_result("D", "OUT_D last lines", 10),
    _a_text("Logs are clean.", 11),
]


def _write(path, lines, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _turn_with(turns, needle):
    hits = [t for t in turns if needle in t["assistant_text"]]
    assert len(hits) == 1, (needle, [t["assistant_text"] for t in turns])
    return hits[0]


def test_result_lands_with_the_call(tmp_path):
    path = str(tmp_path / "s.jsonl")
    _write(path, SESSION)
    turns = list(parse_jsonl_file(path))
    assert "OUT_A" in _turn_with(turns, "ls /backup")["tool_result_text"]
    assert "OUT_A" not in _turn_with(turns, "cat monitor.sh")["tool_result_text"]
    assert "OUT_D" in _turn_with(turns, "tail log")["tool_result_text"]
    assert _turn_with(turns, "Logs are clean.")["tool_result_text"] == ""


def test_parallel_results_both_kept(tmp_path):
    path = str(tmp_path / "s.jsonl")
    _write(path, SESSION)
    turns = list(parse_jsonl_file(path))
    step = _turn_with(turns, "cat monitor.sh")
    assert "OUT_B" in step["tool_result_text"]
    assert "OUT_C" in step["tool_result_text"]


def test_turn_texts_and_indexes_unchanged_by_attribution(tmp_path):
    """Embeddings come from user_text + assistant_text, so these must not move."""
    path = str(tmp_path / "s.jsonl")
    _write(path, SESSION)
    turns = list(parse_jsonl_file(path))
    assert [(t["message_index"], t["user_text"], t["assistant_text"]) for t in turns] == [
        (0, "check the backup", "[Tool: Bash] ls /backup"),
        (1, "", "[Tool: Bash] cat monitor.sh\n\n[Tool: Bash] df -h"),
        (2, "", "Backup is fine."),
        (3, "thanks, now the logs", "[Tool: Bash] tail log"),
        (4, "", "Logs are clean."),
    ]


def test_turn_waiting_for_results_at_eof_is_provisional(tmp_path):
    path = str(tmp_path / "s.jsonl")
    _write(path, SESSION[:5])  # B and C called, no results yet
    turns = list(parse_jsonl_file(path))
    last = turns[-1]
    assert last["provisional"] and "cat monitor.sh" in last["assistant_text"]

    _write(path, SESSION[:6])  # B answered, C still pending
    last = list(parse_jsonl_file(path))[-1]
    assert last["provisional"] and "OUT_B" in last["tool_result_text"]


class FakeModel:
    def embed(self, texts):
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            v[hash(t) % 384] = 1.0
            yield v


def _table(conn):
    return conn.execute(
        """SELECT c.message_index, c.split_index, c.chunk_text, c.tool_result_text, f.tool_result_text
           FROM chunks c JOIN chunks_fts f ON f.rowid = c.id
           ORDER BY c.message_index, c.split_index"""
    ).fetchall()


def _full_table(tmp_path):
    path = str(tmp_path / "full" / "s.jsonl")
    os.makedirs(os.path.dirname(path))
    _write(path, SESSION)
    conn = init_db(str(tmp_path / "full" / "t.db"))
    index_file(conn, FakeModel(), path, "p")
    table = _table(conn)
    conn.close()
    return table


@pytest.mark.parametrize("cut", range(1, len(SESSION)))
def test_incremental_index_matches_full_run(tmp_path, cut):
    expected = _full_table(tmp_path)
    work = tmp_path / f"cut{cut}"
    work.mkdir()
    path = str(work / "s.jsonl")
    conn = init_db(str(work / "t.db"))
    try:
        _write(path, SESSION[:cut])
        index_file(conn, FakeModel(), path, "p")
        _write(path, SESSION[cut:], mode="a")
        index_file(conn, FakeModel(), path, "p")
        assert _table(conn) == expected
    finally:
        conn.close()


def _v1_index(tmp_path, name="v1"):
    """An index whose tool_result_text is shifted one turn later, as parser v1 wrote it."""
    work = tmp_path / name
    work.mkdir()
    path = str(work / "s.jsonl")
    _write(path, SESSION)
    conn = init_db(str(work / "t.db"))
    index_file(conn, FakeModel(), path, "p")
    rows = conn.execute("SELECT id, chunk_text, tool_result_text FROM chunks ORDER BY message_index").fetchall()
    shifted = [""] + [r[2] for r in rows[:-1]]
    for (cid, text, _), tool in zip(rows, shifted):
        conn.execute("UPDATE chunks SET tool_result_text = ? WHERE id = ?", (tool, cid))
        conn.execute("INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                     (cid, text, tool))
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'parser_version'")
    conn.commit()
    return conn


def test_new_index_starts_at_parser_v2(tmp_path):
    from deja.db import get_meta
    conn = init_db(str(tmp_path / "t.db"))
    assert get_meta(conn)["parser_version"] == "2"
    conn.close()


def test_reattach_fixes_v1_index_without_touching_vectors(tmp_path):
    from deja.db import get_meta
    from deja.indexer import reattach_tool_results
    expected = _full_table(tmp_path)
    conn = _v1_index(tmp_path)
    try:
        vec_before = conn.execute("SELECT rowid, embedding FROM chunks_vec ORDER BY rowid").fetchall()
        assert _table(conn) != expected
        stats = reattach_tool_results(conn, FakeModel())
        assert stats["reindexed"] == 0 and stats["updated"] > 0
        assert _table(conn) == expected
        assert conn.execute("SELECT rowid, embedding FROM chunks_vec ORDER BY rowid").fetchall() == vec_before
        assert get_meta(conn)["parser_version"] == "2"
        assert reattach_tool_results(conn, FakeModel())["files"] == 0
    finally:
        conn.close()


def test_reattach_reindexes_session_whose_text_differs(tmp_path):
    from deja.indexer import reattach_tool_results
    expected = _full_table(tmp_path)
    conn = _v1_index(tmp_path)
    try:
        conn.execute("UPDATE chunks SET chunk_text = 'stale' WHERE message_index = 2")
        conn.commit()
        stats = reattach_tool_results(conn, FakeModel())
        assert stats["reindexed"] == 1
        assert _table(conn) == expected
    finally:
        conn.close()


def _m(entry, msg_id):
    entry["message"]["id"] = msg_id
    return entry


# One response (msg M1) streams calls and results interleaved, as Claude Code
# writes parallel tool calls: call F, call G, result F, call H, result G, ...
INTERLEAVED = [
    _u_text("search three things", 0),
    _m(_a_tool("F", "search one", 1), "M1"),
    _m(_a_tool("G", "search two", 2), "M1"),
    _u_result("F", "RES_F", 3),
    _m(_a_tool("H", "search three", 4), "M1"),
    _u_result("G", "RES_G", 5),
    _u_result("H", "RES_H", 6),
    _m(_a_text("All three found.", 7), "M2"),
    _u_text("next", 8),
    _m(_a_text("Done.", 9), "M3"),
]


def test_interleaved_results_follow_their_call_ids(tmp_path):
    path = str(tmp_path / "s.jsonl")
    _write(path, INTERLEAVED)
    turns = list(parse_jsonl_file(path))
    first = _turn_with(turns, "search two")
    assert "RES_F" in first["tool_result_text"] and "RES_G" in first["tool_result_text"]
    third = _turn_with(turns, "search three")
    assert third["tool_result_text"] == "RES_H"
    assert [t["message_index"] for t in turns] == list(range(len(turns)))


@pytest.mark.parametrize("cut", range(1, len(INTERLEAVED)))
def test_incremental_interleaved_matches_full_run(tmp_path, cut):
    full = tmp_path / "full"
    full.mkdir()
    _write(str(full / "s.jsonl"), INTERLEAVED)
    conn = init_db(str(full / "t.db"))
    index_file(conn, FakeModel(), str(full / "s.jsonl"), "p")
    expected = _table(conn)
    conn.close()

    work = tmp_path / "inc"
    work.mkdir()
    path = str(work / "s.jsonl")
    conn = init_db(str(work / "t.db"))
    try:
        _write(path, INTERLEAVED[:cut])
        index_file(conn, FakeModel(), path, "p")
        _write(path, INTERLEAVED[cut:], mode="a")
        index_file(conn, FakeModel(), path, "p")
        assert _table(conn) == expected
    finally:
        conn.close()
