import os
import sys
from itertools import islice
from fastembed import TextEmbedding
from fastembed.text.text_embedding import PoolingType, ModelSource
from deja.db import serialize_f32
from deja.parsers.registry import get_parser
from deja.chunker import make_chunks

EMBED_BATCH_SIZE = 32
TURNS_PER_BATCH = 50

def get_embedding_model() -> TextEmbedding:
    try:
        TextEmbedding.add_custom_model(
            model="intfloat/multilingual-e5-small",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf="intfloat/multilingual-e5-small"),
            dim=384,
            model_file="onnx/model.onnx",
        )
    except ValueError:
        pass  # already registered
    return TextEmbedding(model_name="intfloat/multilingual-e5-small")

def check_needs_reindex(conn, path: str) -> bool | str:
    row = conn.execute(
        "SELECT last_offset, last_mtime, last_size FROM indexed_files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return "full"

    last_offset, last_mtime, last_size = row
    stat = os.stat(path)
    current_size = stat.st_size
    current_mtime = stat.st_mtime

    if current_size < last_offset:
        return "full"
    if current_mtime != last_mtime and current_size == last_size:
        return "full"
    if current_mtime == last_mtime and current_size == last_size:
        return False
    return "incremental"

def _delete_file_chunks(conn, session_id: str):
    chunk_ids = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ?", (session_id,)
    ).fetchall()
    for (cid,) in chunk_ids:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    # analytics pre-aggregates reset on full reindex (incremental tolerates drift)
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM tool_calls WHERE session_id = ?", (session_id,))


def _upsert_session(conn, session_id: str, source: str, project_path: str, turn: dict,
                    kind: str = "main"):
    usage = turn.get("usage") or {}
    ts = turn.get("timestamp", "")
    conn.execute(
        """INSERT INTO sessions (
              session_id, source, kind, project_path, started_at, ended_at, turn_count,
              input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens
           )
           VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
              project_path = COALESCE(sessions.project_path, excluded.project_path),
              started_at = CASE
                  WHEN sessions.started_at IS NULL OR excluded.started_at < sessions.started_at
                  THEN excluded.started_at ELSE sessions.started_at END,
              ended_at = CASE
                  WHEN sessions.ended_at IS NULL OR excluded.ended_at > sessions.ended_at
                  THEN excluded.ended_at ELSE sessions.ended_at END,
              turn_count = sessions.turn_count + 1,
              input_tokens = sessions.input_tokens + excluded.input_tokens,
              output_tokens = sessions.output_tokens + excluded.output_tokens,
              cache_creation_tokens = sessions.cache_creation_tokens + excluded.cache_creation_tokens,
              cache_read_tokens = sessions.cache_read_tokens + excluded.cache_read_tokens
        """,
        (
            session_id, source, kind, project_path, ts, ts,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(usage.get("cache_creation_tokens", 0) or 0),
            int(usage.get("cache_read_tokens", 0) or 0),
        ),
    )


def _upsert_tool_calls(conn, session_id: str, tool_names: list):
    for name in tool_names or []:
        if not name:
            continue
        conn.execute(
            """INSERT INTO tool_calls (session_id, tool_name, call_count) VALUES (?, ?, 1)
               ON CONFLICT(session_id, tool_name) DO UPDATE SET call_count = call_count + 1""",
            (session_id, name),
        )

def _delete_chunks_from(conn, session_id: str, from_index: int):
    ids = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ? AND message_index >= ?",
        (session_id, from_index),
    ).fetchall()
    for (cid,) in ids:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
    conn.execute(
        "DELETE FROM chunks WHERE session_id = ? AND message_index >= ?",
        (session_id, from_index),
    )

def _get_resume_state(conn, session_id: str, path: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT last_offset, next_message_index FROM indexed_files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return 0, 0
    offset = row[0]
    if row[1] is not None:
        return offset, row[1]
    # legacy rows without next_message_index: derive from chunks
    row2 = conn.execute(
        "SELECT MAX(message_index) FROM chunks WHERE session_id = ?", (session_id,)
    ).fetchone()
    start_idx = (row2[0] + 1) if row2 and row2[0] is not None else 0
    return offset, start_idx

def _iter_batches(iterator, size):
    """Yield lists of up to `size` items from iterator."""
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch

def index_file(
    conn,
    model: TextEmbedding,
    path: str,
    project_path: str,
    source: str = "claude-code",
    kind: str = "main",
):
    session_id = os.path.splitext(os.path.basename(path))[0]
    needs = check_needs_reindex(conn, path)

    if needs == False:
        return

    offset = 0
    start_message_index = 0

    if needs == "full":
        _delete_file_chunks(conn, session_id)
    elif needs == "incremental":
        offset, start_message_index = _get_resume_state(conn, session_id, path)
        _delete_chunks_from(conn, session_id, start_message_index)

    stat_before = os.stat(path)
    parser = get_parser(source)
    turns_gen = parser.parse(path, offset=offset, start_message_index=start_message_index)
    indexed_any = False

    for batch_turns in _iter_batches(turns_gen, TURNS_PER_BATCH):
        chunks = []
        for turn in batch_turns:
            chunks.extend(
                make_chunks(turn, session_id, project_path, source=source, kind=kind)
            )
            if not turn.get("provisional"):
                _upsert_session(conn, session_id, source, project_path, turn, kind=kind)
                _upsert_tool_calls(conn, session_id, turn.get("tool_names") or [])

        if not chunks:
            continue

        # Embed in sub-batches
        texts = [c["chunk_text"] for c in chunks]
        all_embeddings = []
        for emb_start in range(0, len(texts), EMBED_BATCH_SIZE):
            emb_batch = texts[emb_start:emb_start + EMBED_BATCH_SIZE]
            all_embeddings.extend(model.embed(emb_batch))

        for chunk, embedding in zip(chunks, all_embeddings):
            try:
                _upsert_chunk(conn, chunk, embedding)
            except Exception as e:
                print(f"[deja] error inserting chunk: {e}", file=sys.stderr)

        # Commit after each batch — crash-safe resume from last committed offset
        last = batch_turns[-1]
        batch_offset = last.get("completed_offset", None)
        next_idx = last["message_index"] if last.get("provisional") else last["message_index"] + 1
        if batch_offset is not None:
            _update_file_meta(conn, path, batch_offset, source=source, next_message_index=next_idx, stat_result=stat_before)
        conn.commit()
        indexed_any = True

    if not indexed_any:
        _update_file_meta(conn, path, offset, source=source, next_message_index=start_message_index, stat_result=stat_before)
        conn.commit()

def _upsert_chunk(conn, chunk: dict, embedding):
    vec_bytes = serialize_f32(embedding.tolist())

    row = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ? AND message_index = ? AND split_index = ?",
        (chunk["session_id"], chunk["message_index"], chunk["split_index"]),
    ).fetchone()

    source = chunk.get("source", "claude-code")
    kind = chunk.get("kind", "main")
    git_branch = chunk.get("git_branch")
    if row:
        chunk_id = row[0]
        conn.execute(
            """UPDATE chunks SET timestamp = ?, project_path = ?,
               chunk_text = ?, tool_result_text = ?, source = ?, kind = ?, git_branch = ?
               WHERE id = ?""",
            (chunk["timestamp"], chunk["project_path"],
             chunk["chunk_text"], chunk.get("tool_result_text", ""),
             source, kind, git_branch, chunk_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (chunk_id, vec_bytes),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
            (chunk_id, chunk["chunk_text"], chunk.get("tool_result_text", "")),
        )
    else:
        cursor = conn.execute(
            """INSERT INTO chunks
            (session_id, message_index, split_index, timestamp, project_path, chunk_text, tool_result_text, source, kind, git_branch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk["session_id"], chunk["message_index"], chunk["split_index"],
             chunk["timestamp"], chunk["project_path"], chunk["chunk_text"],
             chunk.get("tool_result_text", ""), source, kind, git_branch),
        )
        chunk_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (chunk_id, vec_bytes),
        )
        conn.execute(
            "INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
            (chunk_id, chunk["chunk_text"], chunk.get("tool_result_text", "")),
        )

def _update_file_meta(conn, path: str, completed_offset: int = None,
                      source: str = "claude-code", next_message_index: int = None,
                      stat_result=None):
    stat = stat_result if stat_result is not None else os.stat(path)
    offset = completed_offset if completed_offset is not None else stat.st_size
    conn.execute(
        """INSERT OR REPLACE INTO indexed_files
           (path, last_offset, last_mtime, last_size, source, next_message_index)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (path, offset, stat.st_mtime, stat.st_size, source, next_message_index),
    )

def gc_orphans(conn, known_paths: set[str], sources: list[str] = None):
    """Remove index entries for files that no longer exist on disk.

    `sources` limits gc to files indexed from those sources — a partial run
    (`deja index --source codex`) must not treat other sources' files as orphans.
    """
    if sources is None:
        indexed = conn.execute("SELECT path FROM indexed_files").fetchall()
    else:
        placeholders = ",".join("?" * len(sources))
        indexed = conn.execute(
            f"SELECT path FROM indexed_files WHERE source IN ({placeholders})",
            sources,
        ).fetchall()
    for (path,) in indexed:
        if path not in known_paths:
            session_id = os.path.splitext(os.path.basename(path))[0]
            _delete_file_chunks(conn, session_id)
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (path,))
            print(f"[deja] gc: removed orphan {path}", file=sys.stderr)
    conn.commit()
