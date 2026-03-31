import json
import os
import tempfile
from deja.parser import parse_jsonl_file, extract_content

def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def test_extract_text_content():
    content = [{"type": "text", "text": "Hello world"}]
    text, tool_text = extract_content(content)
    assert text == "Hello world"
    assert tool_text == ""

def test_extract_tool_use():
    content = [
        {"type": "text", "text": "Let me check"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
    ]
    text, tool_text = extract_content(content)
    assert "Let me check" in text
    assert "[Tool: Bash] ls -la" in text

def test_extract_tool_result_separate():
    content = [
        {"type": "tool_result", "content": "total 42\ndrwxr-xr-x 2 user user 4096 file.txt"}
    ]
    text, tool_text = extract_content(content)
    assert text == ""
    assert "total 42" in tool_text

def test_extract_skips_thinking():
    content = [
        {"type": "thinking", "thinking": "Let me think..."},
        {"type": "text", "text": "Here is the answer"},
    ]
    text, tool_text = extract_content(content)
    assert "think" not in text.lower()
    assert "Here is the answer" in text

def test_parse_jsonl_file_extracts_turns():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        _write_jsonl(path, [
            {"type": "summary", "summary": "Test session"},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "How to fix nginx?"}]},
                "timestamp": "2026-03-30T10:00:00Z",
                "uuid": "msg-001",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Check the config file."}]},
                "timestamp": "2026-03-30T10:00:05Z",
                "uuid": "msg-002",
            },
        ])
        turns = list(parse_jsonl_file(path))
        assert len(turns) == 1
        assert "nginx" in turns[0]["user_text"]
        assert "config" in turns[0]["assistant_text"]
        assert turns[0]["timestamp"] == "2026-03-30T10:00:05Z"

def test_parse_jsonl_skips_malformed_lines():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broken.jsonl")
        with open(path, "w") as f:
            f.write('{"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"}\n')
            f.write("NOT VALID JSON\n")
            f.write('{"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"}\n')
        turns = list(parse_jsonl_file(path))
        assert len(turns) == 1

def test_parse_jsonl_with_offset():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        lines = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "first"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp1"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "second"}]}, "timestamp": "2026-01-01T00:00:02Z", "uuid": "3"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "resp2"}]}, "timestamp": "2026-01-01T00:00:03Z", "uuid": "4"},
        ]
        _write_jsonl(path, lines)
        all_turns = list(parse_jsonl_file(path, offset=0))
        assert len(all_turns) == 2
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            f.readline()
            offset = f.tell()
        partial_turns = list(parse_jsonl_file(path, offset=offset))
        assert len(partial_turns) == 1
        assert "second" in partial_turns[0]["user_text"]

def test_tool_result_truncated_to_2000():
    long_result = "x" * 5000
    content = [{"type": "tool_result", "content": long_result}]
    text, tool_text = extract_content(content)
    assert len(tool_text) <= 2000
