from typing import Iterator, Protocol


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

    def discover(self) -> Iterator[tuple[str, str]]:
        """Yield (file_path, project_path) for every session file on disk."""
        ...

    def parse(
        self, path: str, offset: int = 0, start_message_index: int = 0
    ) -> Iterator[dict]:
        ...
