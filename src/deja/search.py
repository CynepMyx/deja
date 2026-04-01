import math
import sqlite3
from datetime import datetime, timezone

from deja.db import serialize_f32

TIME_DECAY_ALPHA = 0.98

def fts5_escape(query: str) -> str:
    """Escape query for FTS5: token-wise AND, each token quoted."""
    tokens = query.split()
    if not tokens:
        return '""'
    escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " AND ".join(escaped)

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

def _apply_time_decay(results: list[dict], alpha: float = TIME_DECAY_ALPHA) -> list[dict]:
    now = datetime.now(timezone.utc)
    for r in results:
        ts = r.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            days_ago = max((now - dt).total_seconds() / 86400, 0)
            r["score"] *= alpha ** math.log1p(days_ago)
        except (ValueError, TypeError):
            pass
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
    time_decay: bool = False,
) -> list[dict]:
    has_filters = project or date_from or date_to
    k = 100 if has_filters else 20
    vec_results = _vector_search(conn, model, query, k=k)
    fts_results = _fts_search(conn, query, k=k)
    merged = _rrf_merge(vec_results, fts_results)

    if time_decay:
        merged = _apply_time_decay(merged)

    if project:
        merged = [r for r in merged if r.get("project_path") == project]
    if date_from:
        merged = [r for r in merged if r.get("timestamp", "") >= date_from]
    if date_to:
        merged = [r for r in merged if r.get("timestamp", "") <= date_to]

    return merged[:limit]
