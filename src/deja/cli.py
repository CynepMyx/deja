import os
import sys
import glob
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from deja.db import init_db, get_meta, SCHEMA_VERSION
from deja.indexer import get_embedding_model, index_file, gc_orphans

DEFAULT_INDEX_DIR = os.path.join(
    os.path.expanduser("~"),
    ".claude", "deja",
)
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_INDEX_DIR, "index.db")
LOCK_PATH = os.path.join(DEFAULT_INDEX_DIR, "index.lock")

CLAUDE_PROJECTS_DIR = os.path.join(os.path.expanduser("~"), ".claude", "projects")

def _acquire_lock():
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    lock_fd = open(LOCK_PATH, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("[deja] another indexer is running, exiting", file=sys.stderr)
        sys.exit(1)
    return lock_fd

def _release_lock(lock_fd):
    lock_fd.close()
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass

def _find_jsonl_files() -> list[tuple[str, str]]:
    results = []
    if not os.path.isdir(CLAUDE_PROJECTS_DIR):
        print(f"[deja] {CLAUDE_PROJECTS_DIR} not found", file=sys.stderr)
        return results

    for project_dir in os.listdir(CLAUDE_PROJECTS_DIR):
        full_project = os.path.join(CLAUDE_PROJECTS_DIR, project_dir)
        if not os.path.isdir(full_project):
            continue
        for jsonl in glob.glob(os.path.join(full_project, "*.jsonl")):
            results.append((jsonl, project_dir))

    return results

def cmd_index(args):
    lock_fd = _acquire_lock()
    try:
        os.makedirs(DEFAULT_INDEX_DIR, exist_ok=True)
        conn = init_db(DEFAULT_INDEX_PATH)

        meta = get_meta(conn)
        if args.reindex or int(meta.get("schema_version", "0")) != SCHEMA_VERSION:
            print("[deja] full reindex requested", file=sys.stderr)
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_vec")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM indexed_files")
            conn.commit()

        print("[deja] loading embedding model...", file=sys.stderr)
        model = get_embedding_model()

        files = _find_jsonl_files()
        print(f"[deja] found {len(files)} JSONL files", file=sys.stderr)

        known_paths = set()
        for i, (path, project) in enumerate(files):
            known_paths.add(path)
            print(f"[deja] [{i+1}/{len(files)}] {os.path.basename(path)}", file=sys.stderr)
            index_file(conn, model, path, project)

        gc_orphans(conn, known_paths)
        conn.close()
        print("[deja] indexing complete", file=sys.stderr)
    finally:
        _release_lock(lock_fd)

def cmd_stats():
    import sqlite3
    import sqlite_vec
    from datetime import datetime

    if not os.path.exists(DEFAULT_INDEX_PATH):
        print("[deja] index not found. Run 'deja index' first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DEFAULT_INDEX_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vectors = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    files = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM chunks").fetchone()[0]
    projects = conn.execute("SELECT COUNT(DISTINCT project_path) FROM chunks").fetchone()[0]

    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())

    db_size = os.path.getsize(DEFAULT_INDEX_PATH) / 1024 / 1024
    db_mtime = datetime.fromtimestamp(os.path.getmtime(DEFAULT_INDEX_PATH)).strftime("%Y-%m-%d %H:%M")

    # Consistency check
    issues = []
    if chunks != vectors:
        issues.append(f"chunks ({chunks}) != vectors ({vectors})")
    if chunks != fts:
        issues.append(f"chunks ({chunks}) != fts ({fts})")

    orphans = conn.execute(
        "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    if orphans:
        issues.append(f"{orphans} orphan vector rows")

    print(f"Chunks:     {chunks:,}")
    print(f"Vectors:    {vectors:,}")
    print(f"FTS:        {fts:,}")
    print(f"Sessions:   {sessions}")
    print(f"Projects:   {projects}")
    print(f"Files:      {files}")
    print(f"Model:      {meta.get('embedding_model', '?')}")
    print(f"Dim:        {meta.get('embedding_dim', '?')}")
    print(f"Schema:     v{meta.get('schema_version', '?')}")
    print(f"DB size:    {db_size:.1f} MB")
    print(f"DB path:    {DEFAULT_INDEX_PATH}")
    print(f"Last index: {db_mtime}")

    if issues:
        print(f"\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nHealth:     OK")

    conn.close()

def cmd_serve(args):
    from deja.server import mcp
    mcp.run(transport="stdio")

def cmd_search(args):
    import sqlite3
    import sqlite_vec
    from deja.indexer import get_embedding_model
    from deja.search import hybrid_search

    if not os.path.exists(DEFAULT_INDEX_PATH):
        print("[deja] index not found. Run 'deja index' first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DEFAULT_INDEX_PATH, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    print("Loading model...", file=sys.stderr)
    model = get_embedding_model()

    results = hybrid_search(conn, model, args.query, limit=args.limit, project=args.project)

    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            sid = r.get("session_id", "?")[:12]
            ts = r.get("timestamp", "")[:19]
            text = r.get("chunk_text", "")[:200].replace("\n", " ")
            print(f"\n[{i}] score={score:.4f} | {sid} | {ts}")
            print(f"    {text}")

    conn.close()

def cmd_redact():
    import sqlite3
    import sqlite_vec
    from deja.secrets import redact

    if not os.path.exists(DEFAULT_INDEX_PATH):
        print("[deja] index not found.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DEFAULT_INDEX_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    rows = conn.execute("SELECT id, chunk_text, tool_result_text FROM chunks").fetchall()
    updated = 0
    for row_id, chunk_text, tool_text in rows:
        new_chunk = redact(chunk_text)
        new_tool = redact(tool_text) if tool_text else tool_text
        if new_chunk != chunk_text or new_tool != tool_text:
            conn.execute(
                "UPDATE chunks SET chunk_text = ?, tool_result_text = ? WHERE id = ?",
                (new_chunk, new_tool, row_id),
            )
            conn.execute(
                "INSERT OR REPLACE INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, ?)",
                (row_id, new_chunk, new_tool or ""),
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"[deja] redacted {updated} chunks (embeddings unchanged)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(prog="deja", description="Semantic search for Claude Code sessions")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index JSONL session files")
    idx.add_argument("--reindex", action="store_true", help="Force full reindex")

    sub.add_parser("serve", help="Start MCP server (stdio)")

    sub.add_parser("stats", help="Show index statistics")

    sr = sub.add_parser("search", help="Search indexed sessions")
    sr.add_argument("query", help="Search query")
    sr.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    sr.add_argument("--project", default=None, help="Filter by project path")

    sub.add_parser("redact", help="Redact secrets in existing index (no re-embedding)")

    ev = sub.add_parser("eval", help="Evaluate search quality with golden pairs")
    ev.add_argument("--golden", default=None, help="Path to golden_pairs.json")
    ev.add_argument("--limit", type=int, default=5, help="Results per query (default: 5)")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "stats":
        cmd_stats()
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "redact":
        cmd_redact()
    elif args.command == "eval":
        from deja.eval import evaluate
        evaluate(golden_path=args.golden, limit=args.limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
