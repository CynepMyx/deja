import glob
import json
import os
import sys
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

def extract_content(content) -> tuple[str, str]:
    if isinstance(content, str):
        return content, ""

    text_parts = []
    tool_result_parts = []

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "tool_use":
            name = block.get("name", "unknown")
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

    return "\n".join(text_parts), "\n".join(tool_result_parts)

def parse_jsonl_file(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    pending_user = None
    asst_parts: list[str] = []
    asst_tools: list[str] = []
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

            if entry_type == "user":
                if pending_user is not None and asst_parts:
                    yield _build(line_start)
                    message_index += 1
                text, tool_text = extract_content(content)
                pending_user = {
                    "text": text,
                    "tool_result": tool_text,
                    "timestamp": timestamp,
                }
                asst_parts = []
                asst_tools = []
                last_ts = ""
                turn_start = line_start

            elif entry_type == "assistant":
                if pending_user is None:
                    continue
                text, tool_text = extract_content(content)
                if text:
                    asst_parts.append(text)
                if tool_text:
                    asst_tools.append(tool_text)
                last_ts = timestamp

        if pending_user is not None and asst_parts:
            yield _build(turn_start, provisional=True)

def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()


parse = parse_jsonl_file
