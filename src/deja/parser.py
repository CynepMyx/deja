import json
import sys
from typing import Generator

TOOL_RESULT_MAX = 2000

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
                tool_result_parts.append(raw[:TOOL_RESULT_MAX])

        elif block_type == "thinking":
            continue

    return "\n".join(text_parts), "\n".join(tool_result_parts)

def parse_jsonl_file(
    path: str, offset: int = 0
) -> Generator[dict, None, None]:
    pending_user = None
    message_index = 0

    with open(path, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)

        for line in f:
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
                text, tool_text = extract_content(content)
                pending_user = {
                    "text": text,
                    "tool_result": tool_text,
                    "timestamp": timestamp,
                }

            elif entry_type == "assistant" and pending_user is not None:
                text, tool_text = extract_content(content)
                combined_tool = "\n".join(
                    filter(None, [pending_user["tool_result"], tool_text])
                )
                yield {
                    "user_text": pending_user["text"],
                    "assistant_text": text,
                    "tool_result_text": combined_tool[:TOOL_RESULT_MAX],
                    "timestamp": timestamp or pending_user["timestamp"],
                    "message_index": message_index,
                }
                message_index += 1
                pending_user = None

def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()
