import math
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone

from deja.db import serialize_f32

TIME_DECAY_ALPHA = 0.98
# Filtered searches rank exact distances over at most this many candidates;
# wider filters fall back to KNN over the whole index with a larger k.
PREFILTER_MAX_CANDIDATES = 20_000
_VEC_BATCH = 500
_KNN_MAX_K = 4096
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CHUNK_COLS = """c.id, c.session_id, c.message_index, c.timestamp,
               c.project_path, c.chunk_text, c.tool_result_text"""


def fts5_escape(query: str) -> str:
    """Escape query for FTS5: token-wise AND, each token quoted.

    Tokens with no word character ("/", "-") are dropped: the tokenizer
    yields nothing for them, and an empty phrase would make the AND match
    no rows at all.
    """
    tokens = [t for t in query.split() if re.search(r"\w", t)]
    if not tokens:
        return '""'
    escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " AND ".join(escaped)


def _filter_clause(project=None, source=None, git_branch=None,
                   git_branch_prefix=None, date_from=None, date_to=None) -> tuple[str, list]:
    """SQL conditions on `chunks c` for the search filters.

    A date-only date_to (YYYY-MM-DD) includes that whole day; a full
    timestamp is compared as given. Returns ("", []) when no filter is set.
    """
    clauses, params = [], []
    if project:
        clauses.append("c.project_path = ?")
        params.append(project)
    if source:
        clauses.append("c.source = ?")
        params.append(source)
    if git_branch:
        clauses.append("c.git_branch = ?")
        params.append(git_branch)
    if git_branch_prefix:
        clauses.append("substr(c.git_branch, 1, length(?)) = ?")
        params += [git_branch_prefix, git_branch_prefix]
    if date_from:
        clauses.append("c.timestamp >= ?")
        params.append(date_from)
    if date_to:
        if _DATE_ONLY.match(date_to):
            next_day = date.fromisoformat(date_to) + timedelta(days=1)
            clauses.append("c.timestamp < ?")
            params.append(next_day.isoformat())
        else:
            clauses.append("c.timestamp <= ?")
            params.append(date_to)
    return " AND ".join(clauses), params


def _row_to_dict(r, extra_key):
    return {
        "id": r[0], "session_id": r[1], "message_index": r[2],
        "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
        "tool_result_text": r[6], extra_key: r[7],
    }


def _knn(conn, query_bytes, k, where, params):
    rows = conn.execute(
        f"""
        WITH vec_results AS (
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
        )
        SELECT {_CHUNK_COLS}, v.distance
        FROM vec_results v
        JOIN chunks c ON c.id = v.rowid
        {"WHERE " + where if where else ""}
        ORDER BY v.distance
        """,
        (query_bytes, k, *params),
    ).fetchall()
    return [_row_to_dict(r, "distance") for r in rows]


def _exact_over(conn, query_bytes, ids, k):
    """Exact L2 distance over the given chunk ids (vec0's default metric)."""
    best = []
    for i in range(0, len(ids), _VEC_BATCH):
        batch = ids[i:i + _VEC_BATCH]
        ph = ",".join("?" * len(batch))
        best += conn.execute(
            f"""SELECT rowid, vec_distance_l2(embedding, ?) AS d
                FROM chunks_vec WHERE rowid IN ({ph}) ORDER BY d LIMIT ?""",
            (query_bytes, *batch, k),
        ).fetchall()
    best.sort(key=lambda r: r[1])
    best = best[:k]
    if not best:
        return []
    dist = dict(best)
    ph = ",".join("?" * len(dist))
    rows = conn.execute(
        f"SELECT {_CHUNK_COLS} FROM chunks c WHERE c.id IN ({ph})", list(dist)
    ).fetchall()
    results = [_row_to_dict((*r, dist[r[0]]), "distance") for r in rows]
    results.sort(key=lambda r: r["distance"])
    return results


def _vector_search(conn, model, query: str, k: int = 20,
                   where: str = "", params=()) -> list[dict]:
    query_embedding = list(model.embed([query]))[0]
    query_bytes = serialize_f32(query_embedding.tolist())
    if not where:
        return _knn(conn, query_bytes, k, "", ())

    matched = conn.execute(
        f"SELECT COUNT(*) FROM chunks c WHERE {where}", params
    ).fetchone()[0]
    if matched == 0:
        return []
    if matched <= PREFILTER_MAX_CANDIDATES:
        ids = [r[0] for r in conn.execute(f"SELECT c.id FROM chunks c WHERE {where}", params)]
        return _exact_over(conn, query_bytes, ids, k)

    # Wide filter: most of the index matches, so KNN with k scaled by the
    # filter's selectivity still leaves about k rows after filtering.
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    knn_k = min(_KNN_MAX_K, max(k, math.ceil(k * total / matched)))
    return _knn(conn, query_bytes, knn_k, where, params)[:k]


def _fts_search(conn, query: str, k: int = 20,
                where: str = "", params=()) -> list[dict]:
    escaped = fts5_escape(query)
    try:
        rows = conn.execute(
            f"""
            SELECT {_CHUNK_COLS}, rank
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ? {"AND " + where if where else ""}
            ORDER BY rank
            LIMIT ?
            """,
            (escaped, *params, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [_row_to_dict(r, "fts_rank") for r in rows]

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
    """Hydrate each result with source + git_branch from chunks."""
    if not results:
        return results
    ids = [r["id"] for r in results]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, source, git_branch FROM chunks WHERE id IN ({placeholders})", ids
    ).fetchall()
    meta = {row[0]: (row[1], row[2]) for row in rows}
    for r in results:
        src, branch = meta.get(r["id"], ("claude-code", None))
        r["source"] = src
        r["git_branch"] = branch
    return results


# Backwards-compat alias for any external callers.
_annotate_source = _annotate_metadata


def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
    source: str = None, git_branch: str = None, git_branch_prefix: str = None,
    time_decay: bool = False,
) -> list[dict]:
    where, params = _filter_clause(project, source, git_branch, git_branch_prefix,
                                   date_from, date_to)
    k = 100 if where else 20
    vec_results = _vector_search(conn, model, query, k=k, where=where, params=params)
    fts_results = _fts_search(conn, query, k=k, where=where, params=params)
    merged = _rrf_merge(vec_results, fts_results)
    merged = _annotate_metadata(conn, merged)

    if time_decay:
        merged = _apply_time_decay(merged)

    return merged[:limit]
