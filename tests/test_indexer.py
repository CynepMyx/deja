import json
import os
import time
import tempfile
from deja.db import init_db, serialize_f32
from deja.indexer import index_file, get_embedding_model, check_needs_reindex, gc_orphans

def _make_session(tmp, filename, lines):
    path = os.path.join(tmp, filename)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path

def _append_lines(path, lines):
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def test_index_file_inserts_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "test-project")
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count >= 1
        vec_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        assert vec_count >= 1
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_count >= 1
        conn.close()

def test_incremental_index_skips_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "test"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        needs = check_needs_reindex(conn, path)
        assert needs == False
        conn.close()

def test_safe_reindex_on_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "test"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        conn.execute(
            "UPDATE indexed_files SET last_offset = 99999 WHERE path = ?",
            (path,),
        )
        conn.commit()
        needs = check_needs_reindex(conn, path)
        assert needs == "full"
        conn.close()

def test_incremental_append_no_collision():
    """New turns appended to file get correct message_index; completed turns keep stable ids.

    Note: the LAST turn of a file is provisional (re-upserted fuller on next run),
    so only chunks below next_message_index are guaranteed id-stable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        # Run 1: turn 0 completes (user2 follows it), turn 1 is the provisional tail
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "first question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "first answer"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "second question"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "second answer"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])
        index_file(conn, model, path, "proj")
        turn0_ids = {r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE message_index = 0").fetchall()}
        assert turn0_ids, "turn 0 must be indexed"

        # Run 2: append turn 2; turn 1 (was provisional) is re-upserted, turn 0 untouched
        time.sleep(0.1)
        _append_lines(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "third question"}]}, "timestamp": "2026-01-01T00:02:00Z", "uuid": "5"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "third answer"}]}, "timestamp": "2026-01-01T00:02:01Z", "uuid": "6"},
        ])
        index_file(conn, model, path, "proj")

        turn0_ids_after = {r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE message_index = 0").fetchall()}
        assert turn0_ids_after == turn0_ids, "Completed turns below next_message_index must keep stable ids"

        all_texts = [r[0] for r in conn.execute(
            "SELECT chunk_text FROM chunks ORDER BY message_index").fetchall()]
        assert any("first" in t for t in all_texts)
        assert any("second" in t for t in all_texts)
        assert any("third" in t for t in all_texts)

        indices = [r[0] for r in conn.execute(
            "SELECT DISTINCT message_index FROM chunks ORDER BY message_index").fetchall()]
        assert indices == [0, 1, 2], f"Expected message indices [0, 1, 2], got {indices}"

        conn.close()

def test_dangling_user_not_lost():
    """If file ends with user message (no assistant yet), that turn is picked up on next run."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        # Write complete turn + dangling user
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "complete question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "complete answer"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "dangling question"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
        ])
        index_file(conn, model, path, "proj")
        count1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count1 >= 1  # only the complete turn

        # Now assistant arrives
        time.sleep(0.1)
        _append_lines(path, [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "dangling answer"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])

        index_file(conn, model, path, "proj")
        count2 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count2 >= 2, f"Expected at least 2 chunks after dangling resolved, got {count2}"

        all_texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks").fetchall()]
        assert any("dangling" in t for t in all_texts), "Dangling turn should now be indexed"

        conn.close()

def test_row_consistency_chunks_vec_fts():
    """chunks, chunks_vec, and chunks_fts row counts must match after indexing."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()

        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q1"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a1"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")

        # Append and reindex
        time.sleep(0.1)
        _append_lines(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q2"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a2"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "4"},
        ])
        index_file(conn, model, path, "proj")

        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vec_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

        assert chunks_count == vec_count, f"chunks={chunks_count} != vec={vec_count}"
        assert chunks_count == fts_count, f"chunks={chunks_count} != fts={fts_count}"

        # No orphan rowids in vec
        orphan_vec = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT id FROM chunks)"
        ).fetchone()[0]
        assert orphan_vec == 0, f"Found {orphan_vec} orphan vec rows"

        conn.close()

def test_gc_orphans_scoped_to_sources():
    """Partial run (--source codex) must not treat other sources' files as orphans."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "cc-sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj", source="claude-code")
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 1

        # gc scoped to codex: claude-code file is NOT in known_paths but must survive
        gc_orphans(conn, set(), sources=["codex"])
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 1

        # gc scoped to claude-code: now it is a real orphan and gets removed
        gc_orphans(conn, set(), sources=["claude-code"])
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 0

        conn.close()

def test_gc_orphans_unscoped_removes_all():
    """sources=None keeps the old global behaviour (full `deja index` run)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "cc-sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj", source="claude-code")

        gc_orphans(conn, set())
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 0
        conn.close()

def test_file_with_only_dangling_user_indexed_later():
    """File containing only a user message must not be marked fully indexed (C1)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "lonely question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
        ])
        index_file(conn, model, path, "proj")
        # offset must be 0, not file size
        row = conn.execute("SELECT last_offset FROM indexed_files WHERE path = ?", (path,)).fetchone()
        assert row[0] == 0, f"Expected offset 0 for unindexed dangling user, got {row[0]}"

        time.sleep(0.1)
        _append_lines(path, [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "late answer"}]}, "timestamp": "2026-01-01T00:00:05Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks").fetchall()]
        assert any("lonely" in t for t in texts), "Dangling-only turn must be indexed after assistant arrives"
        conn.close()


def _codex_user(text, ts="2026-06-01T10:00:00Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]}}

def _codex_asst(text, ts="2026-06-01T10:00:05Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}

def test_codex_live_session_tail_grows_without_duplicates():
    """Provisional last turn is re-upserted fuller on next run (B3)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "rollout-x.jsonl", [
            {"type": "session_meta", "payload": {"cwd": "/proj"}},
            _codex_user("question one"),
            _codex_asst("answer part one"),
        ])
        index_file(conn, model, path, "/proj", source="codex")
        texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks ORDER BY message_index").fetchall()]
        assert any("part one" in t for t in texts)

        time.sleep(0.1)
        _append_lines(path, [
            _codex_asst("answer part two"),
            _codex_user("question two", ts="2026-06-01T11:00:00Z"),
            _codex_asst("answer two", ts="2026-06-01T11:00:05Z"),
        ])
        index_file(conn, model, path, "/proj", source="codex")

        rows = conn.execute(
            "SELECT message_index, chunk_text FROM chunks ORDER BY message_index, split_index"
        ).fetchall()
        turn0 = " ".join(t for i, t in rows if i == 0)
        assert "part one" in turn0 and "part two" in turn0, "Tail of live turn must be re-indexed"
        assert any("answer two" in t for _, t in rows), "Next turn must be indexed"
        pairs = [r for r in conn.execute(
            "SELECT message_index, split_index FROM chunks ORDER BY message_index, split_index"
        ).fetchall()]
        assert pairs == sorted(set(pairs)), f"Duplicate (message_index, split_index) rows: {pairs}"
        assert {p[0] for p in pairs} == {0, 1}, f"Expected exactly turns 0 and 1, got {pairs}"
        vec_count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunks_count == vec_count == fts_count, \
            f"Row drift: chunks={chunks_count} vec={vec_count} fts={fts_count}"
        conn.close()


def test_provisional_turn_shrink_clears_stale_splits():
    """_delete_chunks_from must remove stale split rows that re-index will not overwrite."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        # long answer -> multiple split chunks for message_index 0
        path = _make_session(tmp, "rollout-y.jsonl", [
            {"type": "session_meta", "payload": {"cwd": "/proj"}},
            _codex_user("long question"),
            _codex_asst("A" * 4000),
        ])
        index_file(conn, model, path, "/proj", source="codex")
        splits_before = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE message_index = 0").fetchone()[0]
        assert splits_before > 1, "fixture must produce multiple splits"

        # Inject a stale split row at split_index=splits_before (beyond what re-index produces).
        # Without _delete_chunks_from this row survives the re-upsert because _upsert_chunk
        # only touches rows it actually produces — it never deletes extras.
        session_id = os.path.splitext(os.path.basename(path))[0]
        cur = conn.execute(
            "INSERT INTO chunks (session_id, message_index, split_index, timestamp,"
            " project_path, chunk_text, tool_result_text, source)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (session_id, 0, splits_before, "2026-06-01T10:00:05Z", "/proj", "STALE", "", "codex"),
        )
        cid = cur.lastrowid
        conn.execute("INSERT INTO chunks_vec (rowid, embedding) VALUES (?,?)",
                     (cid, serialize_f32([0.0] * 384)))
        conn.execute("INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?,?,?)",
                     (cid, "STALE", ""))
        conn.commit()

        # grow the file with a new turn so the incremental path fires and turn 0 is re-upserted
        time.sleep(0.1)
        _append_lines(path, [
            _codex_user("next q", ts="2026-06-01T11:00:00Z"),
            _codex_asst("short", ts="2026-06-01T11:00:05Z"),
        ])
        index_file(conn, model, path, "/proj", source="codex")

        stale = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE message_index = 0 AND split_index = ?",
            (splits_before,),
        ).fetchone()[0]
        assert stale == 0, "stale split row must be cleared by _delete_chunks_from"
        splits_after = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE message_index = 0").fetchone()[0]
        assert splits_after == splits_before, "turn 0 splits must be exactly re-created, no stale extras"
        orphan_vec = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT id FROM chunks)").fetchone()[0]
        assert orphan_vec == 0, f"{orphan_vec} orphan vector rows after provisional re-upsert"
        conn.close()
