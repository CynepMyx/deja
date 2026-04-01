import os
import sys
from itertools import islice
from fastembed import TextEmbedding
from fastembed.text.text_embedding import PoolingType, ModelSource
from deja.db import serialize_f32
from deja.parser import parse_jsonl_file
from deja.chunker import make_chunks
from deja.secrets import redact

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

def _get_resume_state(conn, session_id: str, path: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT last_offset FROM indexed_files WHERE path = ?", (path,)
    ).fetchone()
    offset = row[0] if row else 0

    row = conn.execute(
        "SELECT MAX(message_index) FROM chunks WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    start_idx = (row[0] + 1) if row and row[0] is not None else 0

    return offset, start_idx

def _iter_batches(iterator, size):
    """Yield lists of up to `size` items from iterator."""
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch

def index_file(conn, model: TextEmbedding, path: str, project_path: str):
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

    turns_gen = parse_jsonl_file(path, offset=offset, start_message_index=start_message_index)
    indexed_any = False

    for batch_turns in _iter_batches(turns_gen, TURNS_PER_BATCH):
        chunks = []
        for turn in batch_turns:
            chunks.extend(make_chunks(turn, session_id, project_path))

        if not chunks:
            continue

        # Embed in sub-batches
        texts = [c["chunk_text"] for c in chunks]
        all_embeddings = []
        for emb_start in range(0, len(texts), EMBED_BATCH_SIZE):
            emb_batch = texts[emb_start:emb_start + EMBED_BATCH_SIZE]
            all_embeddings.extend(model.embed(emb_batch))

        for chunk, embedding in zip(chunks, all_embeddings):
            chunk["chunk_text"] = redact(chunk["chunk_text"])
            if chunk.get("tool_result_text"):
                chunk["tool_result_text"] = redact(chunk["tool_result_text"])
            try:
                _upsert_chunk(conn, chunk, embedding)
            except Exception as e:
                print(f"[deja] error inserting chunk: {e}", file=sys.stderr)

        # Commit after each batch — crash-safe resume from last committed offset
        batch_offset = batch_turns[-1].get("completed_offset", None)
        if batch_offset:
            _update_file_meta(conn, path, batch_offset)
        conn.commit()
        indexed_any = True

    if not indexed_any:
        _update_file_meta(conn, path, offset)
        conn.commit()

def _upsert_chunk(conn, chunk: dict, embedding):
    vec_bytes = serialize_f32(embedding.tolist())

    row = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ? AND message_index = ? AND split_index = ?",
        (chunk["session_id"], chunk["message_index"], chunk["split_index"]),
    ).fetchone()

    if row:
        chunk_id = row[0]
        conn.execute(
            """UPDATE chunks SET timestamp = ?, project_path = ?,
               chunk_text = ?, tool_result_text = ?
               WHERE id = ?""",
            (chunk["timestamp"], chunk["project_path"],
             chunk["chunk_text"], chunk.get("tool_result_text", ""), chunk_id),
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
            (session_id, message_index, split_index, timestamp, project_path, chunk_text, tool_result_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk["session_id"], chunk["message_index"], chunk["split_index"],
             chunk["timestamp"], chunk["project_path"], chunk["chunk_text"],
             chunk.get("tool_result_text", "")),
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

def _update_file_meta(conn, path: str, completed_offset: int = None):
    stat = os.stat(path)
    offset = completed_offset if completed_offset else stat.st_size
    conn.execute(
        """INSERT OR REPLACE INTO indexed_files (path, last_offset, last_mtime, last_size)
        VALUES (?, ?, ?, ?)""",
        (path, offset, stat.st_mtime, stat.st_size),
    )

def gc_orphans(conn, known_paths: set[str]):
    indexed = conn.execute("SELECT path FROM indexed_files").fetchall()
    for (path,) in indexed:
        if path not in known_paths:
            session_id = os.path.splitext(os.path.basename(path))[0]
            _delete_file_chunks(conn, session_id)
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (path,))
            print(f"[deja] gc: removed orphan {path}", file=sys.stderr)
    conn.commit()
