import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from deja.db import open_db_readonly, get_meta, SCHEMA_VERSION
from deja.indexer import get_embedding_model
from deja.search import hybrid_search

DEFAULT_INDEX_PATH = os.path.join(
    os.path.expanduser("~"),
    ".claude", "deja", "index.db",
)


def _check_schema(conn):
    meta = get_meta(conn)
    db_version = int(meta.get("schema_version", "0"))
    if db_version != SCHEMA_VERSION:
        raise ToolError(
            f"Index schema version mismatch: expected {SCHEMA_VERSION}, got {db_version}. "
            "Run 'deja index --reindex' to rebuild."
        )


@asynccontextmanager
async def lifespan(server):
    index_path = os.environ.get("DEJA_INDEX_PATH", DEFAULT_INDEX_PATH)
    if not os.path.exists(index_path):
        print(f"[deja] index not found at {index_path}, search will fail", file=sys.stderr)
        yield {"model": None, "db": None}
        return

    print("[deja] loading model...", file=sys.stderr)
    model = await asyncio.to_thread(get_embedding_model)
    db = open_db_readonly(index_path)
    _check_schema(db)
    print("[deja] ready", file=sys.stderr)
    yield {"model": model, "db": db}
    db.close()


mcp = FastMCP("deja", lifespan=lifespan)


def _do_search(conn, model, query, limit=10, project=None, date_from=None, date_to=None):
    return hybrid_search(conn, model, query, limit=limit,
                         project=project, date_from=date_from, date_to=date_to)


def _do_get_session(conn, session_id):
    rows = conn.execute(
        """SELECT chunk_text, message_index, timestamp, project_path
        FROM chunks WHERE session_id = ? ORDER BY message_index, split_index""",
        (session_id,),
    ).fetchall()
    return [
        {"chunk_text": r[0], "message_index": r[1], "timestamp": r[2], "project_path": r[3]}
        for r in rows
    ]


def _do_get_context(conn, chunk_id, window):
    anchor = conn.execute(
        "SELECT session_id, message_index FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if not anchor:
        return None, []

    session_id, msg_idx = anchor
    lo = msg_idx - window
    hi = msg_idx + window

    rows = conn.execute(
        """SELECT id, chunk_text, message_index, split_index, timestamp, project_path
        FROM chunks
        WHERE session_id = ? AND message_index BETWEEN ? AND ?
        ORDER BY message_index, split_index""",
        (session_id, lo, hi),
    ).fetchall()

    return chunk_id, [
        {
            "id": r[0], "chunk_text": r[1], "message_index": r[2],
            "split_index": r[3], "timestamp": r[4], "project_path": r[5],
            "is_anchor": r[0] == chunk_id,
        }
        for r in rows
    ]


@mcp.tool()
async def search(
    query: str,
    limit: int = 10,
    project: str = None,
    date_from: str = None,
    date_to: str = None,
    ctx: Context = None,
) -> list[dict]:
    """Search past Claude Code sessions by meaning. Returns relevant conversation chunks with context."""
    lc = ctx.lifespan_context
    model = lc.get("model")
    db = lc.get("db")
    if model is None or db is None:
        raise ToolError("Index not loaded. Run 'deja index' first.")
    return await asyncio.to_thread(_do_search, db, model, query, limit, project, date_from, date_to)


@mcp.tool()
async def get_context(chunk_id: int, window: int = 2, ctx: Context = None) -> dict:
    """Get a chunk with surrounding context. Returns the anchor chunk and neighboring turns (±window by message_index) from the same session."""
    lc = ctx.lifespan_context
    db = lc.get("db")
    if db is None:
        raise ToolError("Index not loaded. Run 'deja index' first.")
    anchor_id, chunks = await asyncio.to_thread(_do_get_context, db, chunk_id, window)
    if anchor_id is None:
        raise ToolError(f"Chunk {chunk_id} not found in index.")
    return {"anchor_id": anchor_id, "chunks": chunks}


@mcp.tool()
async def get_session_chunks(session_id: str, ctx: Context = None) -> list[dict]:
    """Get indexed chunks for a session by session_id. Returns chunk_text fragments, not original messages. Long turns may be split with overlap."""
    lc = ctx.lifespan_context
    db = lc.get("db")
    if db is None:
        raise ToolError("Index not loaded. Run 'deja index' first.")
    results = await asyncio.to_thread(_do_get_session, db, session_id)
    if not results:
        raise ToolError(f"Session '{session_id}' not found in index.")
    return results
