import sqlite3
from deja.db import serialize_f32

def fts5_escape(query: str) -> str:
    return '"' + query.replace('"', '""') + '"'

def _vector_search(conn, model, query: str, k: int = 20) -> list[dict]:
    query_embedding = list(model.embed([query]))[0]
    rows = conn.execute(
        """
        WITH vec_results AS (
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        )
        SELECT c.id, c.session_id, c.message_index, c.timestamp,
               c.project_path, c.chunk_text, c.tool_result_text,
               v.distance
        FROM vec_results v
        JOIN chunks c ON c.id = v.rowid
        """,
        (serialize_f32(query_embedding.tolist()), k),
    ).fetchall()

    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "distance": r[7],
        }
        for r in rows
    ]

def _fts_search(conn, query: str, k: int = 20) -> list[dict]:
    escaped = fts5_escape(query)
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.session_id, c.message_index, c.timestamp,
                   c.project_path, c.chunk_text, c.tool_result_text,
                   rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (escaped, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "fts_rank": r[7],
        }
        for r in rows
    ]

def _rrf_merge(vec_results: list, fts_results: list, k: int = 60) -> list[dict]:
    scores = {}
    items = {}

    for rank, item in enumerate(vec_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        items[doc_id] = item

    for rank, item in enumerate(fts_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in items:
            items[doc_id] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        item = items[doc_id]
        item["score"] = scores[doc_id]
        results.append(item)

    return results

def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
) -> list[dict]:
    vec_results = _vector_search(conn, model, query, k=20)
    fts_results = _fts_search(conn, query, k=20)
    merged = _rrf_merge(vec_results, fts_results)

    if project:
        merged = [r for r in merged if r.get("project_path") == project]
    if date_from:
        merged = [r for r in merged if r.get("timestamp", "") >= date_from]
    if date_to:
        merged = [r for r in merged if r.get("timestamp", "") <= date_to]

    return merged[:limit]
