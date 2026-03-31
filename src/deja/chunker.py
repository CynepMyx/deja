MAX_CHUNK_SIZE = 1500
OVERLAP = 200

def _split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        if end >= len(text):
            chunks.append(text[start:])
            break

        search_region = text[end - OVERLAP:end]
        for sep in ["\n\n", ". ", ".\n", "\n"]:
            pos = search_region.rfind(sep)
            if pos != -1:
                end = end - OVERLAP + pos + len(sep)
                break

        chunks.append(text[start:end])
        start = end - OVERLAP
        if start < 0:
            start = 0

    return chunks

def make_chunks(
    turn: dict, session_id: str, project_path: str
) -> list[dict]:
    embed_text = f"{turn['user_text']}\n\n{turn['assistant_text']}"
    parts = _split_text(embed_text)

    return [
        {
            "chunk_text": part,
            "tool_result_text": turn.get("tool_result_text", "") if i == 0 else "",
            "session_id": session_id,
            "message_index": turn["message_index"],
            "split_index": i,
            "timestamp": turn.get("timestamp", ""),
            "project_path": project_path,
        }
        for i, part in enumerate(parts)
    ]
