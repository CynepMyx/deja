import glob
import json
import os
import sys
from typing import Generator, Iterator

from deja.config import CLAUDE_PROJECTS_DIR
from deja.secrets import redact_turn

SOURCE = "claude-code"
TOOL_RESULT_MAX = 2000
SUBAGENT_DIRNAME = "subagents"
KIND_MAIN = "main"
KIND_SUBAGENT = "subagent"


def discover() -> Iterator[tuple[str, str, str]]:
    """Walk ~/.claude/projects, yield (path, project_dir, kind).

    Two layouts live side by side:
      <project>/<session-id>.jsonl                        -> kind "main"
      <project>/<session-id>/subagents/agent-*.jsonl      -> kind "subagent"

    Sub-agent threads carry work the main transcript never sees (delegated
    research, generated code, tool output), so they are indexed too and kept
    addressable via the `kind` column.
    """
    if not os.path.isdir(CLAUDE_PROJECTS_DIR):
        print(f"[deja] {CLAUDE_PROJECTS_DIR} not found", file=sys.stderr)
        return

    for project_dir in os.listdir(CLAUDE_PROJECTS_DIR):
        full_project = os.path.join(CLAUDE_PROJECTS_DIR, project_dir)
        if not os.path.isdir(full_project):
            continue
        for jsonl in glob.glob(os.path.join(full_project, "*.jsonl")):
            yield jsonl, project_dir, KIND_MAIN
        for jsonl in glob.glob(
            os.path.join(full_project, "*", SUBAGENT_DIRNAME, "*.jsonl")
        ):
            yield jsonl, project_dir, KIND_SUBAGENT

def parent_session_id(path: str) -> str | None:
    """Session that spawned a sub-agent thread, read from its path.

    `<project>/<session-id>/subagents/agent-*.jsonl` -> `<session-id>`.
    Returns None for a main transcript.
    """
    subagents_dir, filename = os.path.split(path)
    session_dir, dirname = os.path.split(subagents_dir)
    if dirname != SUBAGENT_DIRNAME:
        return None
    parent = os.path.basename(session_dir)
    return parent or None


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

def parse_jsonl_file(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    pending_user = None
    asst_parts: list[str] = []
    asst_tools: list[str] = []
    asst_tool_names: list[str] = []
    asst_usage: dict = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}
    git_branch: str | None = None
    last_ts = ""
    turn_start = offset
    message_index = start_message_index

    def _build(completed_offset: int, provisional: bool = False) -> dict:
        combined_tool = "\n".join(
            filter(None, [pending_user["tool_result"], *asst_tools])
        )
        turn = {
            "user_text": pending_user["text"],
            "assistant_text": "\n\n".join(asst_parts),
            "tool_result_text": combined_tool,
            "timestamp": last_ts or pending_user["timestamp"],
            "message_index": message_index,
            "completed_offset": completed_offset,
            "git_branch": git_branch,
            "usage": dict(asst_usage),
            "tool_names": list(asst_tool_names),
        }
        if provisional:
            turn["provisional"] = True
        turn = redact_turn(turn)
        turn["tool_result_text"] = turn["tool_result_text"][:TOOL_RESULT_MAX]
        return turn

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
                if pending_user is not None and asst_parts:
                    yield _build(line_start)
                    message_index += 1
                text, tool_text, _ = extract_content(content)
                pending_user = {
                    "text": text,
                    "tool_result": tool_text,
                    "timestamp": timestamp,
                }
                asst_parts = []
                asst_tools = []
                asst_tool_names = []
                asst_usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}
                git_branch = record_branch
                last_ts = ""
                turn_start = line_start

            elif entry_type == "assistant":
                if pending_user is None:
                    continue
                text, tool_text, tool_names = extract_content(content)
                if text:
                    asst_parts.append(text)
                if tool_text:
                    asst_tools.append(tool_text)
                if tool_names:
                    asst_tool_names.extend(tool_names)
                _accumulate_usage(asst_usage, _extract_usage(message))
                if record_branch:
                    git_branch = record_branch
                last_ts = timestamp

        if pending_user is not None and asst_parts:
            yield _build(turn_start, provisional=True)

def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()


parse = parse_jsonl_file
