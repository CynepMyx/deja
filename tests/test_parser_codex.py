import json
import os
import tempfile

from deja.parsers import codex


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _session_meta(cwd="C:\\proj"):
    return {
        "timestamp": "2026-03-28T00:00:00.000Z",
        "type": "session_meta",
        "payload": {
            "id": "01-test",
            "cwd": cwd,
            "originator": "codex_cli_rs",
            "cli_version": "0.116.0",
        },
    }


def _msg(role, text, ts="2026-03-28T00:00:01.000Z"):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
        },
    }


def _function_call(name, args, ts="2026-03-28T00:00:02.000Z"):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "function_call", "name": name, "arguments": args},
    }


def _function_output(output, ts="2026-03-28T00:00:03.000Z"):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "function_call_output", "output": output},
    }


def _event(payload_type="task_started"):
    return {
        "timestamp": "2026-03-28T00:00:00.500Z",
        "type": "event_msg",
        "payload": {"type": payload_type},
    }


def test_basic_user_assistant_pair():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _event(),
            _msg("user", "fix nginx"),
            _msg("assistant", "check the config"),
        ])
        turns = list(codex.parse(path))
        assert len(turns) == 1
        assert "fix nginx" in turns[0]["user_text"]
        assert "check the config" in turns[0]["assistant_text"]
        assert turns[0]["message_index"] == 0


def test_multi_assistant_between_users_kept():
    """Tool loop: user -> asst1 (tool plan) -> tool -> asst2 (answer) -> user2"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _msg("user", "list files"),
            _msg("assistant", "I'll run ls"),
            _function_call("shell", '{"cmd":"ls"}'),
            _function_output("file1\nfile2"),
            _msg("assistant", "found 2 files"),
            _msg("user", "next"),
            _msg("assistant", "done"),
        ])
        turns = list(codex.parse(path))
        assert len(turns) == 2
        first = turns[0]
        assert "I'll run ls" in first["assistant_text"]
        assert "found 2 files" in first["assistant_text"]
        assert "[Tool: shell]" in first["tool_result_text"]
        assert "file1" in first["tool_result_text"]


def test_developer_and_reasoning_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            {"timestamp": "x", "type": "response_item",
             "payload": {"type": "message", "role": "developer",
                         "content": [{"type": "input_text", "text": "system prompt"}]}},
            {"timestamp": "x", "type": "response_item",
             "payload": {"type": "reasoning", "summary": "thinking..."}},
            _msg("user", "hello"),
            _msg("assistant", "hi"),
        ])
        turns = list(codex.parse(path))
        assert len(turns) == 1
        assert "system prompt" not in turns[0]["user_text"]
        assert "thinking" not in turns[0]["assistant_text"]


def test_assistant_before_user_is_dropped():
    """Codex sometimes emits a leading assistant. With no pending user, skip it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _msg("assistant", "stray"),
            _msg("user", "real question"),
            _msg("assistant", "real answer"),
        ])
        turns = list(codex.parse(path))
        assert len(turns) == 1
        assert "stray" not in turns[0]["assistant_text"]


def test_user_without_assistant_not_yielded():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _msg("user", "abandoned"),
        ])
        turns = list(codex.parse(path))
        assert turns == []


def test_resume_from_offset_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _msg("user", "u1"), _msg("assistant", "a1"),
            _msg("user", "u2"), _msg("assistant", "a2"),
            _msg("user", "u3"), _msg("assistant", "a3"),
        ])
        full = list(codex.parse(path))
        assert len(full) == 3

        first_two = []
        gen = codex.parse(path)
        try:
            for i, t in enumerate(gen):
                first_two.append(t)
                if i == 1:
                    break
        finally:
            gen.close()

        resume_offset = first_two[-1]["completed_offset"]
        resume_idx = first_two[-1]["message_index"] + 1
        rest = list(codex.parse(path, offset=resume_offset, start_message_index=resume_idx))

        combined = first_two + rest
        assert [t["message_index"] for t in combined] == [0, 1, 2]
        assert [t["user_text"] for t in combined] == [t["user_text"] for t in full]


def test_tool_output_truncated():
    huge = "x" * 5000
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        _write_jsonl(path, [
            _session_meta(),
            _msg("user", "go"),
            _function_call("shell", "{}"),
            _function_output(huge),
            _msg("assistant", "done"),
        ])
        turns = list(codex.parse(path))
        assert len(turns) == 1
        assert len(turns[0]["tool_result_text"]) <= 2000


def test_discover_walks_date_tree_and_reads_cwd():
    with tempfile.TemporaryDirectory() as tmp:
        sessions_root = os.path.join(tmp, "2026", "03", "28")
        os.makedirs(sessions_root)
        good = os.path.join(sessions_root, "rollout-a.jsonl")
        _write_jsonl(good, [_session_meta(cwd="C:\\my\\project"), _msg("user", "x"), _msg("assistant", "y")])
        idx = os.path.join(tmp, "session_index.jsonl")
        _write_jsonl(idx, [{"some": "stale"}])

        from deja import config
        original = config.CODEX_SESSIONS_DIR
        config.CODEX_SESSIONS_DIR = tmp
        try:
            discovered = list(codex.discover())
        finally:
            config.CODEX_SESSIONS_DIR = original

        paths = [os.path.basename(p) for p, _ in discovered]
        cwds = [c for _, c in discovered]
        assert "rollout-a.jsonl" in paths
        assert "session_index.jsonl" not in paths
        assert "C:\\my\\project" in cwds


def test_discover_handles_missing_dir():
    from deja import config
    original = config.CODEX_SESSIONS_DIR
    config.CODEX_SESSIONS_DIR = "/path/that/does/not/exist/anywhere"
    try:
        assert list(codex.discover()) == []
    finally:
        config.CODEX_SESSIONS_DIR = original


def test_malformed_line_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_session_meta()) + "\n")
            f.write(json.dumps(_msg("user", "hi")) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps(_msg("assistant", "ok")) + "\n")
        turns = list(codex.parse(path))
        assert len(turns) == 1
        assert "hi" in turns[0]["user_text"]
