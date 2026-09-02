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

# vec0 cannot filter on a metadata column, so an excluded kind is filtered
# after the KNN query. Over-fetch and widen until enough rows survive.
OVERFETCH_FACTOR = 4
OVERFETCH_MAX = 2000


def _vector_search(
    conn, model, query: str, k: int = 20, exclude_kind: str = None
) -> list[dict]:
    query_embedding = list(model.embed([query]))[0]
    blob = serialize_f32(query_embedding.tolist())

    fetch = k
    while True:
        neighbours = conn.execute(
            """SELECT rowid, distance FROM chunks_vec
               WHERE embedding MATCH ? AND k = ?
               ORDER BY distance""",
            (blob, fetch),
        ).fetchall()
        if not neighbours:
            return []

        order = {rowid: i for i, (rowid, _) in enumerate(neighbours)}
        distances = dict(neighbours)
        placeholders = ",".join("?" * len(neighbours))
        params: list = list(order)
        kind_clause = ""
        if exclude_kind:
            kind_clause = " AND kind != ?"
            params.append(exclude_kind)

        rows = conn.execute(
            f"""SELECT id, session_id, message_index, timestamp,
                       project_path, chunk_text, tool_result_text
                FROM chunks
                WHERE id IN ({placeholders}){kind_clause}""",
            params,
        ).fetchall()

        # Widening only helps while the KNN still has neighbours to give:
        # a short result means the index is exhausted, not over-filtered.
        exhausted = len(neighbours) < fetch
        if not exclude_kind or len(rows) >= k or exhausted or fetch >= OVERFETCH_MAX:
            break
        fetch = min(fetch * OVERFETCH_FACTOR, OVERFETCH_MAX)

    rows = sorted(rows, key=lambda r: order[r[0]])[:k]
    rows = [(*r, distances[r[0]]) for r in rows]
    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "distance": r[7],
        }
        for r in rows
    ]

def _fts_search(conn, query: str, k: int = 20, exclude_kind: str = None) -> list[dict]:
    escaped = fts5_escape(query)
    kind_clause = "AND c.kind != ?" if exclude_kind else ""
    params = [escaped, exclude_kind, k] if exclude_kind else [escaped, k]
    try:
        rows = conn.execute(
            f"""
            SELECT c.id, c.session_id, c.message_index, c.timestamp,
                   c.project_path, c.chunk_text, c.tool_result_text,
                   rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ? {kind_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params,
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


def _annotate_metadata(conn, results: list[dict]) -> list[dict]:
    """Hydrate each result with source, kind, git_branch and parent session."""
    if not results:
        return results
    ids = [r["id"] for r in results]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""SELECT c.id, c.source, c.kind, c.git_branch, s.parent_session_id
            FROM chunks c
            LEFT JOIN sessions s ON s.session_id = c.session_id
            WHERE c.id IN ({placeholders})""",
        ids,
    ).fetchall()
    meta = {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}
    for r in results:
        src, kind, branch, parent = meta.get(
            r["id"], ("claude-code", "main", None, None)
        )
        r["source"] = src
        r["kind"] = kind
        r["git_branch"] = branch
        r["parent_session_id"] = parent
    return results


# Backwards-compat alias for any external callers.
_annotate_source = _annotate_metadata


def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
    source: str = None, git_branch: str = None, git_branch_prefix: str = None,
    time_decay: bool = False, include_subagents: bool = False,
) -> list[dict]:
    """Hybrid vector + FTS search over indexed turns.

    include_subagents: sub-agent threads are excluded by default so results
    stay on the user-facing conversation. The exclusion is pushed into both
    retrieval lanes rather than applied afterwards — sub-agent threads can
    outnumber main ones several times over, and a post-filter would let them
    consume every candidate slot before the merge.
    """
    has_filters = (
        project or date_from or date_to or source or git_branch or git_branch_prefix
    )
    k = 100 if has_filters else 20
    exclude_kind = None if include_subagents else "subagent"
    vec_results = _vector_search(conn, model, query, k=k, exclude_kind=exclude_kind)
    fts_results = _fts_search(conn, query, k=k, exclude_kind=exclude_kind)
    merged = _rrf_merge(vec_results, fts_results)
    merged = _annotate_metadata(conn, merged)

    if time_decay:
        merged = _apply_time_decay(merged)

    if project:
        merged = [r for r in merged if r.get("project_path") == project]
    if source:
        merged = [r for r in merged if r.get("source") == source]
    if git_branch:
        merged = [r for r in merged if r.get("git_branch") == git_branch]
    if git_branch_prefix:
        merged = [r for r in merged if (r.get("git_branch") or "").startswith(git_branch_prefix)]
    if date_from:
        merged = [r for r in merged if r.get("timestamp", "") >= date_from]
    if date_to:
        merged = [r for r in merged if r.get("timestamp", "") <= date_to]

    return merged[:limit]
