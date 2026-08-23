"""Parser for weekly-review session extracts.

Extracts are markdown digests written by `extract_sessions.py` before the
original transcript is deleted by `cleanupPeriodDays`. They are often the only
surviving copy of an early session, so they are worth indexing on their own.

Format (one file per session):

    # Сессия YYYY-MM-DD (short_id)
    cwd: C:\\Users\\Oleg\\.local\\bin
    Обменов: 65, ~2,525 токенов текста

    ---

    [user] first line
    continuation line

    [assistant] first line
    continuation line

There are no tool results or token counts in this format, only the dialogue.
"""

import glob
import os
import re
import sys
from typing import Generator, Iterator

from deja.config import CLAUDE_PROJECTS_DIR, REVIEW_SESSIONS_DIR
from deja.secrets import redact_turn

SOURCE = "review-extract"

_ROLE_RE = re.compile(r"^\[(user|assistant)\]\s?(.*)$")
_CWD_RE = re.compile(r"^cwd:\s*(.*)$")
_DATE_RE = re.compile(r"^#\s*\S+\s+(\d{4}-\d{2}-\d{2})")


def encode_project_path(cwd: str) -> str:
    """Match the directory name Claude Code derives from a working directory.

    `C:\\Users\\Oleg\\.local\\bin` becomes `C--Users-Oleg--local-bin`. The rule is
    reconstructed, so prefer a real directory under ~/.claude/projects whenever
    one matches; otherwise both sources would count as separate projects.
    """
    if not cwd:
        return ""
    encoded = re.sub(r"[:\\/.]", "-", cwd.rstrip("\\/"))
    candidate = os.path.join(CLAUDE_PROJECTS_DIR, encoded)
    if os.path.isdir(candidate):
        return encoded
    if os.path.isdir(CLAUDE_PROJECTS_DIR):
        lowered = encoded.lower()
        for name in os.listdir(CLAUDE_PROJECTS_DIR):
            if name.lower() == lowered:
                return name
    return encoded


def _read_header(path: str) -> tuple[str, str]:
    """Return (timestamp, project_path) from the first lines of an extract."""
    date = ""
    cwd = ""
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(4):
            line = f.readline()
            if not line:
                break
            if not date:
                m = _DATE_RE.match(line)
                if m:
                    date = m.group(1)
                    continue
            m = _CWD_RE.match(line)
            if m:
                cwd = m.group(1).strip()
    timestamp = f"{date}T00:00:00.000Z" if date else ""
    return timestamp, encode_project_path(cwd)


def discover() -> Iterator[tuple[str, str]]:
    """Yield (path, project_path) for every extract in ~/.claude/reviews/sessions."""
    if not os.path.isdir(REVIEW_SESSIONS_DIR):
        return

    for path in sorted(glob.glob(os.path.join(REVIEW_SESSIONS_DIR, "*.md"))):
        _, project_path = _read_header(path)
        yield path, project_path


def _byte_offsets(lines: list[str]) -> list[int]:
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line.encode("utf-8"))
    return offsets


def parse_extract_file(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    timestamp, _ = _read_header(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"[deja] cannot read {path}: {e}", file=sys.stderr)
        return

    lines = text.splitlines(keepends=True)
    offsets = _byte_offsets(lines)
    file_end = len(text.encode("utf-8"))

    pending_user: list[str] | None = None
    asst_parts: list[str] = []
    current_role: str | None = None
    turn_start = 0
    message_index = start_message_index

    def _build(completed_offset: int) -> dict:
        turn = {
            "user_text": "\n".join(pending_user).strip(),
            "assistant_text": "\n".join(asst_parts).strip(),
            "tool_result_text": "",
            "timestamp": timestamp,
            "message_index": message_index,
            "completed_offset": completed_offset,
            "git_branch": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            },
            "tool_names": [],
        }
        return redact_turn(turn)

    for i, line in enumerate(lines):
        m = _ROLE_RE.match(line.rstrip("\r\n"))
        line_start = offsets[i]

        if m:
            role, first = m.group(1), m.group(2)

            if role == "user":
                # A new user message closes the previous turn, but only once the
                # assistant has answered; consecutive user messages accumulate.
                if pending_user is not None and asst_parts:
                    if turn_start >= offset:
                        yield _build(line_start)
                        message_index += 1
                    pending_user = None
                    asst_parts = []
                if pending_user is None:
                    pending_user = []
                    turn_start = line_start
                if first:
                    pending_user.append(first)
                current_role = "user"
            else:
                if pending_user is None:
                    # Assistant text with no preceding user message: skip it,
                    # the chunker embeds turns, not orphaned replies.
                    current_role = None
                    continue
                if first:
                    asst_parts.append(first)
                current_role = "assistant"
            continue

        if current_role == "user" and pending_user is not None:
            pending_user.append(line.rstrip("\r\n"))
        elif current_role == "assistant":
            asst_parts.append(line.rstrip("\r\n"))

    if pending_user is not None and asst_parts and turn_start >= offset:
        # Extracts are written once and never appended to, so the final turn is
        # complete rather than provisional.
        yield _build(file_end)


def get_file_end_offset(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(0, 2)
        return f.tell()


parse = parse_extract_file
