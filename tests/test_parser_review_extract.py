import os
import tempfile

from deja.parsers import review_extract


EXTRACT = """# Сессия 2026-02-27 (105c1a32)
cwd: C:\\Users\\Oleg\\.local\\bin
Обменов: 2, ~120 токенов текста

---

[user] первый вопрос
вторая строка вопроса

[assistant] первый ответ
вторая строка ответа

[user] второй вопрос

[assistant] второй ответ
"""


def _write(text=EXTRACT, name="2026-02-27_105c1a32_short.md"):
    d = tempfile.mkdtemp()
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def test_parses_two_turns():
    turns = list(review_extract.parse(_write()))
    assert len(turns) == 2
    assert turns[0]["user_text"] == "первый вопрос\nвторая строка вопроса"
    assert turns[0]["assistant_text"] == "первый ответ\nвторая строка ответа"
    assert turns[1]["user_text"] == "второй вопрос"
    assert turns[1]["assistant_text"] == "второй ответ"


def test_message_index_increments_from_start():
    turns = list(review_extract.parse(_write(), start_message_index=5))
    assert [t["message_index"] for t in turns] == [5, 6]


def test_timestamp_is_full_iso():
    turn = next(iter(review_extract.parse(_write())))
    assert turn["timestamp"] == "2026-02-27T00:00:00.000Z"


def test_no_provisional_turns():
    # Extracts are written once, so every turn is final and gets counted
    # into the sessions table.
    turns = list(review_extract.parse(_write()))
    assert all("provisional" not in t for t in turns)


def test_last_turn_offset_is_file_end():
    path = _write()
    turns = list(review_extract.parse(path))
    assert turns[-1]["completed_offset"] == os.path.getsize(path)


def test_offset_skips_already_indexed_turns():
    path = _write()
    first, second = list(review_extract.parse(path))
    resumed = list(
        review_extract.parse(
            path, offset=first["completed_offset"], start_message_index=1
        )
    )
    assert len(resumed) == 1
    assert resumed[0]["user_text"] == second["user_text"]
    assert resumed[0]["message_index"] == 1


def test_empty_fields_for_missing_data():
    turn = next(iter(review_extract.parse(_write())))
    assert turn["tool_result_text"] == ""
    assert turn["tool_names"] == []
    assert turn["usage"]["input_tokens"] == 0
    assert turn["git_branch"] is None


def test_orphan_assistant_is_skipped():
    text = EXTRACT.replace("[user] первый вопрос\nвторая строка вопроса\n\n", "", 1)
    turns = list(review_extract.parse(_write(text)))
    assert len(turns) == 1
    assert turns[0]["user_text"] == "второй вопрос"


def test_consecutive_user_messages_merge_into_one_turn():
    text = """# Сессия 2026-03-01 (abcdef12)
cwd: C:\\proj

---

[user] первая реплика

[user] вторая реплика

[assistant] ответ
"""
    turns = list(review_extract.parse(_write(text)))
    assert len(turns) == 1
    assert turns[0]["user_text"] == "первая реплика\n\nвторая реплика"


def test_secrets_are_redacted():
    text = EXTRACT.replace(
        "первый ответ", "ключ sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH"
    )
    turn = next(iter(review_extract.parse(_write(text))))
    assert "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH" not in turn["assistant_text"]


def test_discover_reads_header(monkeypatch):
    path = _write()
    monkeypatch.setattr(review_extract, "REVIEW_SESSIONS_DIR", os.path.dirname(path))
    found = list(review_extract.discover())
    assert len(found) == 1
    assert found[0][0] == path


def test_encode_project_path_matches_claude_layout():
    assert (
        review_extract.encode_project_path("C:\\Users\\Oleg\\.local\\bin")
        == "C--Users-Oleg--local-bin"
    )
    assert review_extract.encode_project_path("") == ""


def test_source_is_registered():
    from deja.parsers.registry import all_sources, get_parser

    assert "review-extract" in all_sources()
    assert get_parser("review-extract") is review_extract
