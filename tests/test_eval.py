from deja.eval import is_hit

RESULT = {
    "session_id": "s1",
    "timestamp": "2026-08-02T16:50:13.383Z",
    "chunk_text": "Проверка стоит и работает.",
    "tool_result_text": "резервная копия: /opt/app/scripts/monitor.sh.bak-20260802-164932",
}


def test_session_only_pair_keeps_old_behaviour():
    assert is_hit(RESULT, {"expected_sessions": ["s1", "s2"]})
    assert not is_hit(RESULT, {"expected_sessions": ["s9"]})


def test_expected_text_matches_chunk_or_tool_output_case_insensitive():
    assert is_hit(RESULT, {"expected_text": ["MONITOR.sh.bak-20260802"]})
    assert is_hit(RESULT, {"expected_text": ["нет такого", "проверка стоит"]})
    assert not is_hit(RESULT, {"expected_text": ["нет такого"]})


def test_date_range_is_inclusive_by_day():
    assert is_hit(RESULT, {"expected_text": ["monitor.sh"], "expected_date_from": "2026-08-02",
                           "expected_date_to": "2026-08-02"})
    assert not is_hit(RESULT, {"expected_text": ["monitor.sh"], "expected_date_from": "2026-08-03"})
    assert not is_hit(RESULT, {"expected_text": ["monitor.sh"], "expected_date_to": "2026-08-01"})


def test_all_criteria_must_hold_on_the_same_result():
    assert not is_hit(RESULT, {"expected_sessions": ["s9"], "expected_text": ["monitor.sh"]})
