import os
import sys
from fastembed import TextEmbedding
from fastembed.text.text_embedding import PoolingType, ModelSource
from deja.db import serialize_f32
from deja.parser import parse_jsonl_file, get_file_end_offset
from deja.chunker import make_chunks

BATCH_SIZE = 32

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

def index_file(conn, model: TextEmbedding, path: str, project_path: str):
    session_id = os.path.splitext(os.path.basename(path))[0]
    needs = check_needs_reindex(conn, path)

    if needs == False:
        return

    offset = 0
    if needs == "full":
        _delete_file_chunks(conn, session_id)
    elif needs == "incremental":
        row = conn.execute(
            "SELECT last_offset FROM indexed_files WHERE path = ?", (path,)
        ).fetchone()
        offset = row[0] if row else 0

    turns = list(parse_jsonl_file(path, offset=offset))
    if not turns:
        _update_file_meta(conn, path)
        return

    all_chunks = []
    for turn in turns:
        all_chunks.extend(make_chunks(turn, session_id, project_path))

    texts = [c["chunk_text"] for c in all_chunks]
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        batch_chunks = all_chunks[batch_start:batch_start + BATCH_SIZE]
        embeddings = list(model.embed(batch_texts))

        for chunk, embedding in zip(batch_chunks, embeddings):
            try:
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO chunks
                    (session_id, message_index, split_index, timestamp, project_path, chunk_text, tool_result_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk["session_id"], chunk["message_index"], chunk["split_index"],
                     chunk["timestamp"], chunk["project_path"], chunk["chunk_text"],
                     chunk.get("tool_result_text", "")),
                )
                rowid = cursor.lastrowid
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, serialize_f32(embedding.tolist())),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                    (rowid, chunk["chunk_text"], chunk.get("tool_result_text", "")),
                )
            except Exception as e:
                print(f"[deja] error inserting chunk: {e}", file=sys.stderr)

    _update_file_meta(conn, path)
    conn.commit()

def _update_file_meta(conn, path: str):
    stat = os.stat(path)
    conn.execute(
        """INSERT OR REPLACE INTO indexed_files (path, last_offset, last_mtime, last_size)
        VALUES (?, ?, ?, ?)""",
        (path, stat.st_size, stat.st_mtime, stat.st_size),
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
