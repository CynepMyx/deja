from typing import Iterator, Protocol


def parent_session_id(parser, path: str) -> str | None:
    """Parent session of a delegated thread, or None.

    Optional per parser: sources without a sub-agent concept simply do not
    define `parent_session_id`.
    """
    lookup = getattr(parser, "parent_session_id", None)
    return lookup(path) if lookup else None


class Parser(Protocol):
    """Source-specific session parser.

    Each parser:
    1. Knows where its source stores session files on disk (`discover`).
    2. Knows how to read one file and emit normalized turns (`parse`).

    Turns must be dicts with keys:
        user_text, assistant_text, tool_result_text,
        timestamp, message_index, completed_offset
    """

    SOURCE: str

    def discover(self) -> Iterator[tuple[str, str, str]]:
        """Yield (file_path, project_path, kind) for every session file on disk.

        `kind` is "main" for user-facing sessions and "subagent" for threads
        a session delegated to a sub-agent. Sources without a sub-agent
        concept yield "main" for everything.
        """
        ...

    def parse(
        self, path: str, offset: int = 0, start_message_index: int = 0
    ) -> Iterator[dict]:
        ...
