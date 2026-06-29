import glob
import json
import os
import sys
from typing import Generator, Iterator

from deja import config
from deja.secrets import redact_turn

SOURCE = "codex"
TOOL_RESULT_MAX = 2000
TOOL_ARGS_MAX = 200


def discover() -> Iterator[tuple[str, str]]:
    """Walk ~/.codex/sessions/YYYY/MM/DD/*.jsonl, yield (path, cwd).

    cwd is read from each file's first-line session_meta. We walk the
    filesystem tree directly rather than trusting session_index.jsonl,
    which can drift (per clean-my-agent maintainer's note on #8).
    """
    root = config.CODEX_SESSIONS_DIR
    if not os.path.isdir(root):
        return

    pattern = os.path.join(root, "**", "*.jsonl")
    for path in glob.iglob(pattern, recursive=True):
        if os.path.basename(path) == "session_index.jsonl":
            continue
        cwd = _read_session_cwd(path)
        yield path, cwd


def _read_session_cwd(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            return "unknown"
        entry = json.loads(first)
        if entry.get("type") != "session_meta":
            return "unknown"
        return entry.get("payload", {}).get("cwd", "unknown") or "unknown"
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _extract_message_text(payload: dict) -> str:
    parts = []
    for block in payload.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype in ("input_text", "output_text", "text"):
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def _format_tool_call(payload: dict) -> str:
    name = payload.get("name", "tool")
    args = payload.get("arguments", "")
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return f"[Tool: {name}] {str(args)[:TOOL_ARGS_MAX]}"


def _format_tool_output(payload: dict) -> str:
    output = payload.get("output", "")
    if isinstance(output, dict):
        output = output.get("content") or json.dumps(output, ensure_ascii=False)
    elif isinstance(output, list):
        output = json.dumps(output, ensure_ascii=False)
    return str(output)


def _build_turn(
    user_text: str,
    asst_parts: list[str],
    tool_parts: list[str],
    tool_names: list[str],
    git_branch: str | None,
    timestamp: str,
    message_index: int,
    completed_offset: int,
) -> dict:
    combined_tool = "\n".join(p for p in tool_parts if p)
    turn = {
        "user_text": user_text,
        "assistant_text": "\n\n".join(p for p in asst_parts if p),
        "tool_result_text": combined_tool,
        "timestamp": timestamp,
        "message_index": message_index,
        "completed_offset": completed_offset,
        "git_branch": git_branch,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0},
        "tool_names": list(tool_names),
    }
    turn = redact_turn(turn)
    turn["tool_result_text"] = turn["tool_result_text"][:TOOL_RESULT_MAX]
    return turn


def parse(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    """Parse a Codex rollout JSONL into normalized (user, assistant, tools) turns.

    Pairing strategy: one turn = one user message + every assistant message and
    every function_call / function_call_output up to the next user message.
    This preserves multi-step tool loops (typical of GPT agents) instead of
    discarding all but the first assistant reply.
    """
    pending_user = None
    asst_parts: list[str] = []
    tool_parts: list[str] = []
    tool_names: list[str] = []
    git_branch: str | None = None
    pending_ts = ""
    turn_start = offset
    message_index = start_message_index

    with open(path, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)

        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"[deja] skipping malformed line in {path}", file=sys.stderr)
                continue

            etype = entry.get("type")
            payload = entry.get("payload", {}) or {}

            if etype == "session_meta":
                git_branch = (payload.get("git") or {}).get("branch")
                continue

            if etype != "response_item":
                continue

            ptype = payload.get("type", "")
            role = payload.get("role", "")
            ts = entry.get("timestamp", "")

            if ptype == "message" and role == "user":
                if pending_user is not None and asst_parts:
                    yield _build_turn(
                        pending_user, asst_parts, tool_parts, tool_names,
                        git_branch, pending_ts, message_index, line_start,
                    )
                    message_index += 1
                pending_user = _extract_message_text(payload)
                asst_parts = []
                tool_parts = []
                tool_names = []
                pending_ts = ts
                turn_start = line_start

            elif ptype == "message" and role == "assistant":
                if pending_user is None:
                    continue
                text = _extract_message_text(payload)
                if text:
                    asst_parts.append(text)

            elif ptype in ("function_call", "custom_tool_call"):
                if pending_user is None:
                    continue
                tool_parts.append(_format_tool_call(payload))
                name = payload.get("name")
                if name:
                    tool_names.append(name)

            elif ptype in ("function_call_output", "custom_tool_call_output"):
                if pending_user is None:
                    continue
                tool_parts.append(_format_tool_output(payload))

            # skip everything else: reasoning, web_search_call,
            # developer messages, etc.

        if pending_user is not None and asst_parts:
            turn = _build_turn(
                pending_user, asst_parts, tool_parts, tool_names,
                git_branch, pending_ts, message_index, turn_start,
            )
            turn["provisional"] = True
            yield turn
