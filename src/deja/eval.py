import json
import os
import sys
import time
import sqlite3
import sqlite_vec

from deja.indexer import get_embedding_model
from deja.search import hybrid_search

DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "golden_pairs.json")
DEFAULT_INDEX = os.path.join(os.path.expanduser("~"), ".claude", "deja", "index.db")


def load_golden(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(golden_path: str = None, index_path: str = None, limit: int = 5):
    golden_path = golden_path or DEFAULT_GOLDEN
    index_path = index_path or DEFAULT_INDEX

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
        expected = set(pair["expected_sessions"])

        t0 = time.perf_counter()
        results = hybrid_search(conn, model, query, limit=limit)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        result_sessions = [r["session_id"] for r in results]

        rr = 0.0
        for rank, sid in enumerate(result_sessions, 1):
            if sid in expected:
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
