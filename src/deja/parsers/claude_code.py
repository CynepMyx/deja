import glob
import json
import os
import sys
from collections import deque
from typing import Generator, Iterator

from deja.config import CLAUDE_PROJECTS_DIR
from deja.secrets import redact_turn

SOURCE = "claude-code"
TOOL_RESULT_MAX = 2000


def discover() -> Iterator[tuple[str, str]]:
    """Walk ~/.claude/projects/<project>/*.jsonl, yield (path, project_dir)."""
    if not os.path.isdir(CLAUDE_PROJECTS_DIR):
        print(f"[deja] {CLAUDE_PROJECTS_DIR} not found", file=sys.stderr)
        return

    for project_dir in os.listdir(CLAUDE_PROJECTS_DIR):
        full_project = os.path.join(CLAUDE_PROJECTS_DIR, project_dir)
        if not os.path.isdir(full_project):
            continue
        for jsonl in glob.glob(os.path.join(full_project, "*.jsonl")):
            yield jsonl, project_dir

def extract_content(content) -> tuple[str, str, list[str]]:
    """Returns (text, tool_result, tool_names_used)."""
    if isinstance(content, str):
        return content, "", []

    text_parts = []
    tool_result_parts = []
    tool_names: list[str] = []

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            tool_names.append(name)
            inp = block.get("input", {})
            if isinstance(inp, dict):
                cmd = inp.get("command", inp.get("file_path", inp.get("query", "")))
            else:
                cmd = str(inp)[:200]
            text_parts.append(f"[Tool: {name}] {cmd}")

        elif block_type == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, list):
                raw = " ".join(
                    b.get("text", "") for b in raw if isinstance(b, dict)
                )
            if isinstance(raw, str):
                tool_result_parts.append(raw)

        elif block_type == "thinking":
            continue

    return "\n".join(text_parts), "\n".join(tool_result_parts), tool_names


def _extract_usage(message: dict) -> dict:
    """Extract token counts from message.usage, normalized to a fixed key set."""
    usage = message.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_creation_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
        "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
    }


def _accumulate_usage(acc: dict, new: dict) -> dict:
    for k in ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens"):
        acc[k] = acc.get(k, 0) + new.get(k, 0)
    return acc

def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}


def _tool_results(content) -> list[tuple[str | None, str]]:
    """(tool_use_id, text) for each tool_result block, text as extract_content builds it."""
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if block.get("type") != "tool_result":
            continue
        raw = block.get("content", "")
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") for b in raw if isinstance(b, dict))
        if isinstance(raw, str):
            out.append((block.get("tool_use_id"), raw))
    return out


def _tool_use_ids(content) -> set:
    if not isinstance(content, list):
        return set()
    return {b.get("id") for b in content if b.get("type") == "tool_use" and b.get("id")}


def _build(state: dict, completed_offset: int, provisional: bool = False,
           resume_index: int | None = None) -> dict:
    combined_tool = "\n".join(
        filter(None, [state["user"]["tool_result"], *state["results"], *state["asst_tools"]])
    )
    turn = {
        "user_text": state["user"]["text"],
        "assistant_text": "\n\n".join(state["parts"]),
        "tool_result_text": combined_tool,
        "timestamp": state["last_ts"] or state["user"]["timestamp"],
        "message_index": state["message_index"],
        "completed_offset": completed_offset,
        "git_branch": state["git_branch"],
        "usage": dict(state["usage"]),
        "tool_names": list(state["tool_names"]),
    }
    if provisional:
        turn["provisional"] = True
        turn["resume_message_index"] = (
            state["message_index"] if resume_index is None else resume_index
        )
    turn = redact_turn(turn)
    turn["tool_result_text"] = turn["tool_result_text"][:TOOL_RESULT_MAX]
    return turn


def parse_jsonl_file(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    """Yield one turn per assistant step: the user record before it and the
    assistant entries up to the next user record.

    Tool results arrive as user records after the call, and with parallel
    calls Claude Code interleaves calls and results of one response. A
    finished turn therefore waits in a queue until every tool_use it issued
    has its tool_result (matched by tool_use_id), or until an assistant entry
    of a different message starts, by which point the API has all results.
    Turns leave the queue in order. A turn's completed_offset is the user
    record that closed it, so a resumed parse starts on a user record;
    results there for turns indexed earlier have unknown ids and are skipped.
    Turns still queued at EOF are provisional and restart from the first of
    them (resume_message_index).
    """
    state = None
    queue: deque = deque()
    resumed = offset > 0
    message_index = start_message_index

    def _new_state(user: dict, branch, start: int) -> dict:
        return {
            "user": user, "parts": [], "asst_tools": [], "results": [],
            "tool_names": [], "usage": _empty_usage(), "git_branch": branch,
            "last_ts": "", "start": start, "end": None,
            "message_index": message_index, "ids": set(), "answered": set(),
            "msg_ids": set(),
        }

    def _drain(force_unless_msg=None):
        while queue:
            head = queue[0]
            done = head["ids"] <= head["answered"]
            stale = force_unless_msg is not None and force_unless_msg not in head["msg_ids"]
            if not (done or stale):
                break
            queue.popleft()
            yield _build(head, head["end"])

    with open(path, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)

        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[deja] skipping malformed line in {path}", file=sys.stderr)
                continue

            entry_type = entry.get("type", "")
            if entry_type == "summary":
                continue

            message = entry.get("message", {})
            content = message.get("content", [])
            timestamp = entry.get("timestamp", "")
            record_branch = entry.get("gitBranch")

            if entry_type == "user":
                text, _, _ = extract_content(content)
                if state is not None and state["parts"]:
                    state["end"] = line_start
                    queue.append(state)
                    message_index += 1
                leftover = []
                for tool_id, result in _tool_results(content):
                    owner = next((t for t in reversed(queue) if tool_id in t["ids"]), None)
                    if owner is not None:
                        owner["results"].append(result)
                        owner["answered"].add(tool_id)
                    elif resumed:
                        continue
                    elif queue:
                        queue[-1]["results"].append(result)
                    else:
                        leftover.append(result)
                state = _new_state(
                    {"text": text, "tool_result": "\n".join(leftover), "timestamp": timestamp},
                    record_branch, line_start,
                )
                yield from _drain()

            elif entry_type == "assistant":
                if state is None:
                    continue
                msg_id = message.get("id") or f"line:{line_start}"
                yield from _drain(force_unless_msg=msg_id)
                text, tool_text, tool_names = extract_content(content)
                if text:
                    state["parts"].append(text)
                if tool_text:
                    state["asst_tools"].append(tool_text)
                if tool_names:
                    state["tool_names"].extend(tool_names)
                state["ids"] |= _tool_use_ids(content)
                state["msg_ids"].add(msg_id)
                _accumulate_usage(state["usage"], _extract_usage(message))
                if record_branch:
                    state["git_branch"] = record_branch
                state["last_ts"] = timestamp

        pending = list(queue)
        if state is not None and state["parts"]:
            pending.append(state)
        if pending:
            first = pending[0]
            for t in pending:
                yield _build(t, first["start"], provisional=True,
                             resume_index=first["message_index"])


def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()


parse = parse_jsonl_file
