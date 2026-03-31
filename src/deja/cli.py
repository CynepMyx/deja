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
    ".claude", "projects", "C--Users-Oleg--local-bin", "memory", "deja",
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

def cmd_serve(args):
    from deja.server import mcp
    mcp.run(transport="stdio")

def main():
    parser = argparse.ArgumentParser(prog="deja", description="Semantic search for Claude Code sessions")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index JSONL session files")
    idx.add_argument("--reindex", action="store_true", help="Force full reindex")

    sub.add_parser("serve", help="Start MCP server (stdio)")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
