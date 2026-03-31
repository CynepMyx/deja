from deja.chunker import make_chunks

def test_short_turn_single_chunk():
    turn = {
        "user_text": "How to restart nginx?",
        "assistant_text": "Run: sudo systemctl restart nginx",
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert len(chunks) == 1
    assert "nginx" in chunks[0]["chunk_text"]
    assert chunks[0]["session_id"] == "sess-1"
    assert chunks[0]["split_index"] == 0

def test_long_turn_splits():
    turn = {
        "user_text": "Explain everything about Docker containers.",
        "assistant_text": "A" * 2000,
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["split_index"] == i
        assert len(chunk["chunk_text"]) <= 1700  # 1500 + some overhead

def test_tool_result_not_in_chunk_text():
    turn = {
        "user_text": "Check disk",
        "assistant_text": "Here are the results",
        "tool_result_text": "Filesystem Size Used Avail",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    assert "Filesystem" not in chunks[0]["chunk_text"]
    assert chunks[0]["tool_result_text"] == "Filesystem Size Used Avail"

def test_split_respects_sentence_boundaries():
    sentences = ". ".join([f"Sentence number {i}" for i in range(50)])
    turn = {
        "user_text": "Tell me many things",
        "assistant_text": sentences,
        "tool_result_text": "",
        "timestamp": "2026-03-30T10:00:00Z",
        "message_index": 0,
    }
    chunks = make_chunks(turn, session_id="sess-1", project_path="/proj")
    for chunk in chunks:
        text = chunk["chunk_text"]
        if not text.startswith("Tell me"):
            assert not text[0].islower() or text.startswith(". ")
