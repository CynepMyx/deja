import json
import os
import sys
import time
import sqlite3
import sqlite_vec

from deja.indexer import get_embedding_model
from deja.search import hybrid_search
from deja.config import get_index_path

DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "golden_pairs.json")


def load_golden(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SEARCH_FILTERS = ("project", "source", "git_branch", "git_branch_prefix", "date_from", "date_to")


def is_hit(result: dict, pair: dict) -> bool:
    """Does one search result satisfy a golden pair?

    A pair names what a correct result looks like; every criterion it sets
    must hold for the same result:
      expected_sessions  - the result's session_id is one of these;
      expected_text      - chunk_text or tool_result_text contains one of
                           these strings (case-insensitive);
      expected_date_from / expected_date_to - the result's timestamp falls in
                           this range (YYYY-MM-DD, inclusive).
    Session-level pairs say little when a few long sessions hold most of the
    index; expected_text plus a date range pins the original conversation.
    """
    sessions = pair.get("expected_sessions")
    if sessions and result.get("session_id") not in sessions:
        return False
    texts = pair.get("expected_text")
    if texts:
        hay = " ".join(
            filter(None, [result.get("chunk_text"), result.get("tool_result_text")])
        ).lower()
        if not any(t.lower() in hay for t in texts):
            return False
    day = (result.get("timestamp") or "")[:10]
    if pair.get("expected_date_from") and day < pair["expected_date_from"]:
        return False
    if pair.get("expected_date_to") and day > pair["expected_date_to"]:
        return False
    return True


def evaluate(golden_path: str = None, index_path: str = None, limit: int = 5):
    golden_path = golden_path or DEFAULT_GOLDEN
    index_path = index_path or get_index_path()

    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(index_path, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    print("Loading model...", file=sys.stderr)
    model = get_embedding_model()

    pairs = load_golden(golden_path)
    print(f"Evaluating {len(pairs)} queries @ limit={limit}\n")

    total_mrr = 0.0
    hits = 0
    misses = 0
    total_time = 0.0

    for pair in pairs:
        query = pair["query"]
        filters = {k: v for k, v in (pair.get("filters") or {}).items() if k in SEARCH_FILTERS}

        t0 = time.perf_counter()
        results = hybrid_search(conn, model, query, limit=limit, **filters)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        rr = 0.0
        for rank, result in enumerate(results, 1):
            if is_hit(result, pair):
                rr = 1.0 / rank
                break

        total_mrr += rr

        if rr > 0:
            hits += 1
            status = f"HIT  (rank {int(1/rr)})"
        else:
            misses += 1
            status = "MISS"

        print(f"  {status:14s} {elapsed:.3f}s  {query}")

    mrr = total_mrr / len(pairs)
    avg_time = total_time / len(pairs)

    print(f"\n{'='*50}")
    print(f"MRR@{limit}:        {mrr:.4f}")
    print(f"Hits:          {hits}/{len(pairs)} ({100*hits/len(pairs):.0f}%)")
    print(f"Misses:        {misses}/{len(pairs)}")
    print(f"Avg latency:   {avg_time*1000:.0f}ms")
    print(f"Total time:    {total_time:.1f}s")

    conn.close()
    return mrr
