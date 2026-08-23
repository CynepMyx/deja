# deja: Fix All Review Findings (v0.4.0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть все находки полного ревью 2026-06-13 (`C:\Projects\deja\REVIEW-2026-06-13.md`) и довести ветку `feat/codex-parser` до релиза v0.4.0.

**Architecture:** Вся работа идёт на ветке `feat/codex-parser` (B1/B2 уже починены коммитами `48cac25`, `c45d866`). Схема v2 не зарелизена — любые изменения схемы бесплатны, миграция v1→v2 в `init_db` уже дропает старые таблицы. Ключевые механики: «provisional» turns для живых сессий (B3+m1), редактирование секретов на уровне turn'а до усечения/чанкинга (M1), e5-префиксы + metadata-фильтры в vec0 (M5+M6) — всё в одной миграции v2, один полный реиндекс.

**Tech Stack:** Python 3.10+, sqlite-vec 0.1.8, SQLite FTS5, fastembed (multilingual-e5-small), FastMCP 3.x, pytest.

**Рабочая директория:** `C:\Projects\deja`, venv: `.venv/Scripts/python.exe`. Тесты: `.venv/Scripts/python.exe -m pytest tests/ -q` (полный прогон ~60s, модель кэширована).

**Запреты:** НЕ пушить и НЕ мерджить без явного ок Олега. НЕ запускать `deja index` на реальном индексе (`C:\Users\Oleg\.claude\deja\index.db`) до Task 17 — код ветки при первом `init_db` мигрирует (дропнет) v1-индекс.

**Out of scope (осознанно НЕ делаем):** FTS5 external content (экономия диска — отложено); entropy-based детектор секретов (только префиксные паттерны); reranking #12; parent-child chunking #13; Cursor parser (Phase 2 #8); нормализация project_path между источниками (только документируем).

---

### Task 0: Safety net — бэкап индекса + baseline eval на main-коде

Baseline нужен ДО любых изменений: после M6 (префиксы) сравним MRR. Запускать eval надо кодом main — код ветки упадёт на v1-индексе (нет колонки source).

**Files:** ничего в репо не меняется.

- [ ] **Step 1: Бэкап локального индекса**

```bash
cp /c/Users/Oleg/.claude/deja/index.db /c/Users/Oleg/.claude/deja/index.v1.backup.db
ls -la /c/Users/Oleg/.claude/deja/
```
Expected: копия ~133MB рядом с оригиналом.

- [ ] **Step 2: Worktree с main и baseline eval**

```bash
cd /c/Projects/deja && git worktree add ../deja-main-eval main
cd /c/Projects/deja-main-eval && PYTHONPATH=/c/Projects/deja-main-eval/src /c/Projects/deja/.venv/Scripts/python.exe -m deja.cli eval --golden /c/Projects/deja/tests/golden_pairs.json 2>&1 | tee /c/Projects/deja/docs/plans/eval-baseline.txt
```
Expected: вывод MRR@5, Hits/Misses, латентность. Файл `docs/plans/eval-baseline.txt` сохранён. Записать итоговый MRR.

- [ ] **Step 3: Удалить worktree**

```bash
cd /c/Projects/deja && git worktree remove ../deja-main-eval
```

---

### Task 1: C1 — falsy-0 в `_update_file_meta` теряет первый turn сессии

**Files:**
- Modify: `src/deja/indexer.py` (функция `_update_file_meta`)
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Написать падающий тест**

В конец `tests/test_indexer.py`:

```python
def test_file_with_only_dangling_user_indexed_later():
    """File containing only a user message must not be marked fully indexed (C1)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "lonely question"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
        ])
        index_file(conn, model, path, "proj")
        # offset must be 0, not file size
        row = conn.execute("SELECT last_offset FROM indexed_files WHERE path = ?", (path,)).fetchone()
        assert row[0] == 0, f"Expected offset 0 for unindexed dangling user, got {row[0]}"

        time.sleep(0.1)
        _append_lines(path, [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "late answer"}]}, "timestamp": "2026-01-01T00:00:05Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks").fetchall()]
        assert any("lonely" in t for t in texts), "Dangling-only turn must be indexed after assistant arrives"
        conn.close()
```

- [ ] **Step 2: Прогнать — убедиться что падает**

Run: `.venv/Scripts/python.exe -m pytest tests/test_indexer.py::test_file_with_only_dangling_user_indexed_later -v`
Expected: FAIL на первом assert (offset == размер файла).

- [ ] **Step 3: Фикс — одна строка**

В `src/deja/indexer.py`, в `_update_file_meta` заменить:

```python
    offset = completed_offset if completed_offset else stat.st_size
```
на:
```python
    offset = completed_offset if completed_offset is not None else stat.st_size
```

- [ ] **Step 4: Тест зелёный + весь сьют**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: все passed.

- [ ] **Step 5: Commit**

```bash
git add src/deja/indexer.py tests/test_indexer.py
git commit -m "fix(indexer): file with only a dangling user no longer marked fully indexed

_update_file_meta treated offset=0 as falsy and recorded st_size,
permanently skipping the first turn of a session indexed before the
assistant reply arrived."
```

---

### Task 2: B3 — provisional turns: последний turn живой Codex-сессии

Механика: codex-парсер при EOF выдаёт частичный turn с флагом `provisional: True` и `completed_offset` = начало turn'а. Индексер для provisional turn'а записывает `next_message_index` = его же индекс (не +1), а при инкрементальном resume удаляет чанки с `message_index >= next_message_index` и переиндексирует turn целиком (уже полнее).

**Files:**
- Modify: `src/deja/db.py` (DDL indexed_files: + `next_message_index INTEGER`)
- Modify: `src/deja/parsers/codex.py` (EOF-yield)
- Modify: `src/deja/indexer.py` (`_get_resume_state`, `_update_file_meta`, `index_file`, новый `_delete_chunks_from`)
- Test: `tests/test_parser_codex.py`, `tests/test_indexer.py`

- [ ] **Step 1: Падающий тест парсера**

В `tests/test_parser_codex.py` (используя существующие хелперы фикстур файла; если хелпера записи строк нет — писать json.dumps построчно как в соседних тестах):

```python
def test_eof_turn_is_provisional_with_turn_start_offset(tmp_path):
    path = tmp_path / "rollout.jsonl"
    lines = [
        {"type": "session_meta", "payload": {"cwd": "/proj"}},
        {"type": "response_item", "timestamp": "2026-06-01T10:00:00Z",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "first q"}]}},
        {"type": "response_item", "timestamp": "2026-06-01T10:00:05Z",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "partial answer"}]}},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    turns = list(parse(str(path)))
    assert len(turns) == 1
    assert turns[0].get("provisional") is True
    # re-parse from completed_offset must yield the same turn (offset = turn start)
    again = list(parse(str(path), offset=turns[0]["completed_offset"],
                       start_message_index=turns[0]["message_index"]))
    assert len(again) == 1
    assert again[0]["user_text"] == "first q"
```

- [ ] **Step 2: Прогнать — падает**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parser_codex.py::test_eof_turn_is_provisional_with_turn_start_offset -v`
Expected: FAIL — `provisional` отсутствует, offset = EOF (re-parse пустой).

- [ ] **Step 3: Фикс codex.py**

В `src/deja/parsers/codex.py`, в `parse()`: переименовать `pending_user_line_start` → `turn_start` (объявление `turn_start = offset`, присваивание `turn_start = line_start` в ветке user). Финальный блок заменить на:

```python
        if pending_user is not None and asst_parts:
            turn = _build_turn(
                pending_user, asst_parts, tool_parts,
                pending_ts, message_index, turn_start,
            )
            turn["provisional"] = True
            yield turn
```

- [ ] **Step 4: Тест парсера зелёный**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parser_codex.py -v`
Expected: все PASS (существующий `test_resume_from_offset_is_idempotent` мог опираться на старый offset — если падает, обновить его ожидания под новый контракт: offset последнего turn'а = его начало).

- [ ] **Step 5: Падающий интеграционный тест индексера**

В `tests/test_indexer.py`:

```python
def _codex_user(text, ts="2026-06-01T10:00:00Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]}}

def _codex_asst(text, ts="2026-06-01T10:00:05Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}

def test_codex_live_session_tail_grows_without_duplicates():
    """Provisional last turn is re-upserted fuller on next run (B3)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "rollout-x.jsonl", [
            {"type": "session_meta", "payload": {"cwd": "/proj"}},
            _codex_user("question one"),
            _codex_asst("answer part one"),
        ])
        index_file(conn, model, path, "/proj", source="codex")
        texts = [r[0] for r in conn.execute("SELECT chunk_text FROM chunks ORDER BY message_index").fetchall()]
        assert any("part one" in t for t in texts)

        time.sleep(0.1)
        _append_lines(path, [
            _codex_asst("answer part two"),
            _codex_user("question two", ts="2026-06-01T11:00:00Z"),
            _codex_asst("answer two", ts="2026-06-01T11:00:05Z"),
        ])
        index_file(conn, model, path, "/proj", source="codex")

        rows = conn.execute(
            "SELECT message_index, chunk_text FROM chunks ORDER BY message_index, split_index"
        ).fetchall()
        turn0 = " ".join(t for i, t in rows if i == 0)
        assert "part one" in turn0 and "part two" in turn0, "Tail of live turn must be re-indexed"
        assert any("answer two" in t for _, t in rows), "Next turn must be indexed"
        indices = sorted({i for i, _ in rows})
        assert indices == [0, 1], f"No duplicate/skipped message_index, got {indices}"
        conn.close()
```

- [ ] **Step 6: Прогнать — падает** (turn0 без "part two")

- [ ] **Step 7: Фикс схемы и индексера**

`src/deja/db.py` — в DDL `indexed_files` добавить колонку:

```sql
        CREATE TABLE IF NOT EXISTS indexed_files (
            path TEXT PRIMARY KEY,
            last_offset INTEGER NOT NULL DEFAULT 0,
            last_mtime REAL NOT NULL,
            last_size INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'claude-code',
            next_message_index INTEGER
        );
```

`src/deja/indexer.py`:

```python
def _delete_chunks_from(conn, session_id: str, from_index: int):
    ids = conn.execute(
        "SELECT id FROM chunks WHERE session_id = ? AND message_index >= ?",
        (session_id, from_index),
    ).fetchall()
    for (cid,) in ids:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
    conn.execute(
        "DELETE FROM chunks WHERE session_id = ? AND message_index >= ?",
        (session_id, from_index),
    )

def _get_resume_state(conn, session_id: str, path: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT last_offset, next_message_index FROM indexed_files WHERE path = ?",
        (path,),
    ).fetchone()
    if row is None:
        return 0, 0
    offset = row[0]
    if row[1] is not None:
        return offset, row[1]
    # legacy rows without next_message_index: derive from chunks
    row2 = conn.execute(
        "SELECT MAX(message_index) FROM chunks WHERE session_id = ?", (session_id,)
    ).fetchone()
    start_idx = (row2[0] + 1) if row2 and row2[0] is not None else 0
    return offset, start_idx
```

`_update_file_meta` — новая сигнатура (C1-фикс сохранить!):

```python
def _update_file_meta(conn, path: str, completed_offset: int = None,
                      source: str = "claude-code", next_message_index: int = None):
    stat = os.stat(path)
    offset = completed_offset if completed_offset is not None else stat.st_size
    conn.execute(
        """INSERT OR REPLACE INTO indexed_files
           (path, last_offset, last_mtime, last_size, source, next_message_index)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (path, offset, stat.st_mtime, stat.st_size, source, next_message_index),
    )
```

`index_file` — в ветке incremental добавить очистку provisional-хвоста, в батч-цикле считать `next_idx`:

```python
    elif needs == "incremental":
        offset, start_message_index = _get_resume_state(conn, session_id, path)
        _delete_chunks_from(conn, session_id, start_message_index)
```

```python
        # Commit after each batch — crash-safe resume from last committed offset
        last = batch_turns[-1]
        batch_offset = last.get("completed_offset", None)
        next_idx = last["message_index"] if last.get("provisional") else last["message_index"] + 1
        if batch_offset is not None:
            _update_file_meta(conn, path, batch_offset, source=source, next_message_index=next_idx)
        conn.commit()
        indexed_any = True

    if not indexed_any:
        _update_file_meta(conn, path, offset, source=source, next_message_index=start_message_index)
        conn.commit()
```

- [ ] **Step 8: Всё зелёное**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: все passed (включая Task 1 тест — `_update_file_meta` сохранил `is not None`).

- [ ] **Step 9: Commit**

```bash
git add src/deja/db.py src/deja/parsers/codex.py src/deja/indexer.py tests/test_parser_codex.py tests/test_indexer.py
git commit -m "fix(codex): provisional EOF turn — live session tail is re-indexed, not frozen

The last turn of a growing session was yielded with offset past EOF,
so content appended later was never indexed. EOF turns are now marked
provisional with offset at turn start; indexed_files tracks
next_message_index and incremental resume re-upserts the turn fuller."
```

---

### Task 3: m1 — claude-парсер: аккумулировать подряд идущие assistant-записи (+provisional)

Бэкпорт codex-стратегии: turn открывается user-записью, поглощает ВСЕ последующие assistant-записи, закрывается следующей user-записью (или EOF → provisional). Использует механику Task 2.

**Files:**
- Modify: `src/deja/parsers/claude_code.py` (`parse_jsonl_file`)
- Test: `tests/test_parser.py`

- [ ] **Step 1: Падающий тест**

В `tests/test_parser.py`:

```python
def test_consecutive_assistant_entries_accumulated():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        _write_jsonl(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "do two things"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "first part"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "second part"}]}, "timestamp": "2026-01-01T00:00:02Z", "uuid": "3"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "next"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "4"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "next answer"}]}, "timestamp": "2026-01-01T00:01:01Z", "uuid": "5"},
        ])
        turns = list(parse_jsonl_file(path))
        assert len(turns) == 2
        assert "first part" in turns[0]["assistant_text"]
        assert "second part" in turns[0]["assistant_text"], "Second assistant entry must not be dropped"
        assert turns[1].get("provisional") is True
```

- [ ] **Step 2: Прогнать — падает** ("second part" потерян)

- [ ] **Step 3: Переписать `parse_jsonl_file`**

В `src/deja/parsers/claude_code.py` заменить `parse_jsonl_file` целиком:

```python
def parse_jsonl_file(
    path: str, offset: int = 0, start_message_index: int = 0
) -> Generator[dict, None, None]:
    pending_user = None
    asst_parts: list[str] = []
    asst_tools: list[str] = []
    last_ts = ""
    turn_start = offset
    message_index = start_message_index

    def _build(completed_offset: int, provisional: bool = False) -> dict:
        combined_tool = "\n".join(
            filter(None, [pending_user["tool_result"], *asst_tools])
        )
        turn = {
            "user_text": pending_user["text"],
            "assistant_text": "\n\n".join(asst_parts),
            "tool_result_text": combined_tool[:TOOL_RESULT_MAX],
            "timestamp": last_ts or pending_user["timestamp"],
            "message_index": message_index,
            "completed_offset": completed_offset,
        }
        if provisional:
            turn["provisional"] = True
        return turn

    with open(path, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)

        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break
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
                if pending_user is not None and asst_parts:
                    yield _build(line_start)
                    message_index += 1
                text, tool_text = extract_content(content)
                pending_user = {
                    "text": text,
                    "tool_result": tool_text,
                    "timestamp": timestamp,
                }
                asst_parts = []
                asst_tools = []
                last_ts = ""
                turn_start = line_start

            elif entry_type == "assistant":
                if pending_user is None:
                    continue
                text, tool_text = extract_content(content)
                if text:
                    asst_parts.append(text)
                if tool_text:
                    asst_tools.append(tool_text)
                last_ts = timestamp

        if pending_user is not None and asst_parts:
            yield _build(turn_start, provisional=True)
```

- [ ] **Step 4: Все тесты парсера + сьют**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parser.py tests/test_indexer.py tests/ -q`
Expected: все passed. Если `test_parse_jsonl_file_extracts_turns` падает по timestamp — проверить, что `last_ts` берётся из assistant-записи (ожидание `2026-03-30T10:00:05Z` сохраняется).

- [ ] **Step 5: Commit**

```bash
git add src/deja/parsers/claude_code.py tests/test_parser.py
git commit -m "fix(parser): accumulate consecutive assistant entries into one turn

Claude Code writes one model turn as several assistant JSONL entries;
the old pairing kept only the first and dropped the rest. Backports
the codex accumulate-until-next-user strategy, including provisional
EOF turns for live sessions."
```

---

### Task 4: m2 — гонка «файл вырос во время индексации»

`_update_file_meta` стат-ит файл ПОСЛЕ парсинга: рост между ними записывается как уже проиндексированный. Фикс: stat снимается ДО парсинга и передаётся в meta.

**Files:**
- Modify: `src/deja/indexer.py` (`index_file`, `_update_file_meta`)
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Тест (на сигнатуру/поведение)**

```python
def test_file_meta_uses_preparse_stat():
    """Growth during indexing must be picked up by the next run (m2)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q1"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "a1"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        import os as _os
        stat_before = _os.stat(path)
        # simulate growth AFTER parse but BEFORE meta write: call with explicit stat
        from deja.indexer import _update_file_meta
        time.sleep(0.05)
        _append_lines(path, [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q2"}]}, "timestamp": "2026-01-01T00:01:00Z", "uuid": "3"},
        ])
        _update_file_meta(conn, path, 10, source="claude-code",
                          next_message_index=1, stat_result=stat_before)
        conn.commit()
        row = conn.execute("SELECT last_size FROM indexed_files WHERE path = ?", (path,)).fetchone()
        assert row[0] == stat_before.st_size, "Meta must record pre-parse size so growth triggers reindex"
        conn.close()
```

- [ ] **Step 2: Прогнать — падает** (TypeError: нет параметра stat_result)

- [ ] **Step 3: Фикс**

`_update_file_meta` — добавить параметр:

```python
def _update_file_meta(conn, path: str, completed_offset: int = None,
                      source: str = "claude-code", next_message_index: int = None,
                      stat_result=None):
    stat = stat_result if stat_result is not None else os.stat(path)
    offset = completed_offset if completed_offset is not None else stat.st_size
```
(остальное тело без изменений)

`index_file` — сразу после `needs = check_needs_reindex(conn, path)` / перед парсингом:

```python
    stat_before = os.stat(path)
```
и оба вызова `_update_file_meta(...)` дополнить `stat_result=stat_before`.

- [ ] **Step 4: Сьют зелёный.** Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

- [ ] **Step 5: Commit**

```bash
git add src/deja/indexer.py tests/test_indexer.py
git commit -m "fix(indexer): record pre-parse stat in file meta

Data appended while a file was being indexed was stamped as already
indexed (stat taken after parsing). Meta now records the stat captured
before parsing, so the next run sees the growth."
```

---

### Task 5: M1 — секреты: redact ДО усечения и ДО чанкинга

Редактирование переезжает в парсеры (на уровне turn'а, до усечения tool_result), из индексера убирается. Плюс fallback-паттерн «BEGIN PRIVATE KEY без END».

**Files:**
- Modify: `src/deja/secrets.py` (+`redact_turn`, +fallback-паттерн)
- Modify: `src/deja/parsers/claude_code.py` (extract_content без усечения; redact в `_build`)
- Modify: `src/deja/parsers/codex.py` (то же в `_build_turn` / `_format_tool_output`)
- Modify: `src/deja/indexer.py` (убрать пер-чанковый redact)
- Test: `tests/test_secrets.py`, `tests/test_parser.py`, `tests/test_indexer.py`

- [ ] **Step 1: Падающие тесты**

`tests/test_secrets.py`:

```python
def test_truncated_private_key_redacted():
    text = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "A" * 3000  # END cut off
    result = redact(text)
    assert "AAAA" not in result
    assert REDACTED in result
```

`tests/test_parser.py`:

```python
def test_secret_redacted_before_tool_result_truncation():
    key = "-----BEGIN RSA PRIVATE KEY-----\n" + "B" * 2400 + "\n-----END RSA PRIVATE KEY-----"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        _write_jsonl(path, [
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": key}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        turns = list(parse_jsonl_file(path))
        assert "BBBB" not in turns[0]["tool_result_text"], "Key must be redacted before truncation"
```

`tests/test_indexer.py`:

```python
def test_secret_at_chunk_boundary_not_leaked():
    secret = "password=SuperBoundarySecret99"
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        long_text = "x" * 1495 + " " + secret + " " + "y" * 1500
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "long"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": long_text}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, model, path, "proj")
        for (text,) in conn.execute("SELECT chunk_text FROM chunks").fetchall():
            assert "SuperBoundarySecret" not in text
        conn.close()
```

- [ ] **Step 2: Прогнать — все три падают**

- [ ] **Step 3: Имплементация**

`src/deja/secrets.py` — в конец `PATTERNS` добавить:

```python
    # Truncation fallback: BEGIN header without END still gets redacted
    re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]+'),
```

и новую функцию:

```python
def redact_turn(turn: dict) -> dict:
    turn["user_text"] = redact(turn["user_text"])
    turn["assistant_text"] = redact(turn["assistant_text"])
    if turn.get("tool_result_text"):
        turn["tool_result_text"] = redact(turn["tool_result_text"])
    return turn
```

`src/deja/parsers/claude_code.py`:
- импорт: `from deja.secrets import redact_turn`
- в `extract_content` убрать усечение: `tool_result_parts.append(raw)` (вместо `raw[:TOOL_RESULT_MAX]`)
- в `_build` (из Task 3): убрать `[:TOOL_RESULT_MAX]` из сборки `combined_tool`, в конце:

```python
        turn = redact_turn(turn)
        turn["tool_result_text"] = turn["tool_result_text"][:TOOL_RESULT_MAX]
        return turn
```
(поле в dict собирать как `"tool_result_text": combined_tool` без усечения)

`src/deja/parsers/codex.py`:
- импорт: `from deja.secrets import redact_turn`
- `_format_tool_output`: `return str(output)` (убрать `[:TOOL_RESULT_MAX]`)
- `_build_turn`: `"tool_result_text": combined_tool` без усечения; перед `return`:

```python
    turn = redact_turn(turn)
    turn["tool_result_text"] = turn["tool_result_text"][:TOOL_RESULT_MAX]
    return turn
```

`src/deja/indexer.py`:
- удалить `from deja.secrets import redact`
- удалить из цикла upsert строки:

```python
            chunk["chunk_text"] = redact(chunk["chunk_text"])
            if chunk.get("tool_result_text"):
                chunk["tool_result_text"] = redact(chunk["tool_result_text"])
```

- [ ] **Step 4: Сьют зелёный.** Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
(`test_redact_updates_existing_chunks` обязан остаться зелёным — redact теперь в парсере, но `[REDACTED]` в индексе всё равно появляется.)

- [ ] **Step 5: Commit**

```bash
git add src/deja/secrets.py src/deja/parsers/claude_code.py src/deja/parsers/codex.py src/deja/indexer.py tests/
git commit -m "fix(secrets): redact at turn level, before truncation and chunking

Truncating tool results to 2000 chars before redaction left private
keys without their END marker unredacted; chunk splitting could cut a
secret across the boundary so no pattern matched either half. Turns
are now redacted in the parsers on full text, then truncated. Also
adds a BEGIN-without-END private key fallback pattern. Embeddings are
now computed over redacted text as well."
```

---

### Task 6: M2 — паттерны «голых» токенов

**Files:**
- Modify: `src/deja/secrets.py`
- Test: `tests/test_secrets.py`

- [ ] **Step 1: Падающие тесты**

```python
import pytest

@pytest.mark.parametrize("token", [
    "sk-ant-api03-AbCdEf123456789012345678901234",
    "sk-proj-AbCdEf12345678901234567890",
    "dop_v1_0f1e2d3c4b5a69788796a5b4c3d2e1f009182736",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpM",
    "AIzaSyA-1234567890abcdefghijklmnopqrstuv",
    "sk_live_4eC39HqLyjWDarjtT1zdp7dc12345",
    "1234567890:AAEhBOweik6ad9r_QXMENQjcrGbqCr4K-pc",
    "npm_AbCd1234567890efGhIjKlMnOpQrStUvWx",
])
def test_redact_bare_tokens(token):
    text = f"env output:\n{token}\nnext line"
    result = redact(text)
    assert token not in result, f"Bare token leaked: {token[:12]}..."
    assert REDACTED in result

def test_redact_still_preserves_normal_text():
    text = "Fix the skirt-and-blouse layout for nginx proxy in project alpha"
    assert redact(text) == text
```

- [ ] **Step 2: Прогнать — параметризованные падают**

- [ ] **Step 3: Добавить паттерны в `PATTERNS`**

```python
    # Bare well-known token formats (appear in env dumps / configs without assignment context)
    re.compile(r'\bsk-(?:ant|proj)-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'\b(?:dop_v1_|figd_|npm_)[A-Za-z0-9_\-]{15,}'),
    re.compile(r'\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{15,}'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}'),  # JWT
    re.compile(r'\bAIza[0-9A-Za-z_\-]{35}'),
    re.compile(r'\b[rs]k_live_[A-Za-z0-9]{20,}'),  # Stripe
    re.compile(r'\b\d{8,10}:AA[A-Za-z0-9_\-]{33}'),  # Telegram bot
```

- [ ] **Step 4: Сьют зелёный** (особенно `test_redact_preserves_normal_text` — нет ложных срабатываний)

- [ ] **Step 5: Commit**

```bash
git add src/deja/secrets.py tests/test_secrets.py
git commit -m "feat(secrets): redact bare high-entropy tokens

sk-ant/sk-proj, DigitalOcean, Figma, npm, PyPI, standalone JWTs,
Google API keys, Stripe live keys and Telegram bot tokens are now
caught without assignment context — the common case in tool results
(env dumps, .env reads)."
```

---

### Task 7: M6 — e5-префиксы `passage:` / `query:`

Префикс добавляется ТОЛЬКО в текст для эмбеддинга; в `chunk_text` ничего не меняется.

**Files:**
- Modify: `src/deja/indexer.py` (тексты для embed)
- Modify: `src/deja/search.py` (`_vector_search`)
- Test: `tests/test_indexer.py`, `tests/test_search.py`

- [ ] **Step 1: Падающие тесты со spy-моделью**

В `tests/test_search.py`:

```python
import numpy as np

class _SpyModel:
    def __init__(self):
        self.inputs = []
    def embed(self, texts, **kwargs):
        self.inputs.extend(texts)
        return [np.zeros(384, dtype=np.float32) for _ in texts]

def test_query_embedding_uses_e5_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        spy = _SpyModel()
        hybrid_search(conn, spy, "find nginx config", limit=5)
        assert spy.inputs == ["query: find nginx config"]
        conn.close()
```

В `tests/test_indexer.py`:

```python
def test_passage_prefix_used_for_indexing():
    import numpy as np
    class _SpyModel:
        def __init__(self):
            self.inputs = []
        def embed(self, texts, **kwargs):
            self.inputs.extend(texts)
            return [np.zeros(384, dtype=np.float32) for _ in texts]

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "test.db"))
        spy = _SpyModel()
        path = _make_session(tmp, "sess.jsonl", [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ])
        index_file(conn, spy, path, "proj")
        assert spy.inputs, "embed must be called"
        assert all(t.startswith("passage: ") for t in spy.inputs)
        # stored text must NOT contain the prefix
        row = conn.execute("SELECT chunk_text FROM chunks").fetchone()
        assert not row[0].startswith("passage: ")
        conn.close()
```

- [ ] **Step 2: Прогнать — падают**

- [ ] **Step 3: Имплементация**

`src/deja/indexer.py`, в `index_file`:

```python
        texts = ["passage: " + c["chunk_text"] for c in chunks]
```

`src/deja/search.py`, в `_vector_search`:

```python
    query_embedding = list(model.embed(["query: " + query]))[0]
```

- [ ] **Step 4: Сьют зелёный.** Существующие интеграционные тесты поиска (реальная модель, обе стороны с префиксами) обязаны остаться зелёными.

- [ ] **Step 5: Commit**

```bash
git add src/deja/indexer.py src/deja/search.py tests/
git commit -m "feat(search): e5 query:/passage: prefixes for embeddings

multilingual-e5-small is trained with these prefixes; without them
retrieval quality degrades (per the model card). Prefixes are added
only to embedding inputs, stored chunk_text is unchanged. Requires a
full reindex (already forced by schema v2)."
```

---

### Task 8: M5 — честные фильтры project/source на уровне retrieval

vec0 metadata-колонки (sqlite-vec 0.1.8) для KNN-фильтра + SQL-фильтр в FTS-ветке. `_annotate_source` удаляется (source попадает в SELECT).

**Files:**
- Modify: `src/deja/db.py` (DDL chunks_vec)
- Modify: `src/deja/indexer.py` (`_upsert_chunk` — запись metadata)
- Modify: `src/deja/search.py` (`_vector_search`, `_fts_search`, `hybrid_search`, удалить `_annotate_source`)
- Test: `tests/test_search.py`

- [ ] **Step 1: Спайк — проверить metadata-фильтр в sqlite-vec 0.1.8**

```bash
cd /c/Projects/deja && .venv/Scripts/python.exe - <<'PYEOF'
import sqlite3, sqlite_vec, struct
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)
conn.execute("CREATE VIRTUAL TABLE t USING vec0(embedding float[4], source TEXT)")
pack = lambda v: struct.pack("4f", *v)
conn.execute("INSERT INTO t (rowid, embedding, source) VALUES (1, ?, 'a')", (pack([1,0,0,0]),))
conn.execute("INSERT INTO t (rowid, embedding, source) VALUES (2, ?, 'b')", (pack([0.9,0,0,0]),))
rows = conn.execute(
    "SELECT rowid FROM t WHERE embedding MATCH ? AND k = 2 AND source = 'b'",
    (pack([1,0,0,0]),)).fetchall()
print("metadata filter result:", rows)
assert rows == [(2,)], "metadata filtering not supported"
print("OK: vec0 metadata filtering works")
PYEOF
```
Expected: `OK: vec0 metadata filtering works`.
**Если спайк ПАДАЕТ** (metadata не поддерживается): вместо Step 3-4 применить fallback — в `hybrid_search` заменить `k = 100 if has_filters else 20` на `k = 500 if has_filters else 20`, фильтры оставить пост-hoc, `_annotate_source` всё равно удалить (source добавить в оба SELECT через JOIN c.source). Зафиксировать решение в коммит-месседже.

- [ ] **Step 2: Падающий тест (детерминированный, без реальной модели)**

В `tests/test_search.py`:

```python
from deja.db import serialize_f32
from deja.search import _vector_search

def _insert_chunk(conn, cid, session, project, source, text, vec):
    conn.execute(
        """INSERT INTO chunks (id, session_id, message_index, split_index, timestamp,
           project_path, chunk_text, tool_result_text, source)
           VALUES (?, ?, ?, 0, '2026-01-01T00:00:00Z', ?, ?, '', ?)""",
        (cid, session, cid, project, text, source))
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding, source, project_path) VALUES (?, ?, ?, ?)",
        (cid, serialize_f32(vec), source, project))
    conn.execute(
        "INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, '')",
        (cid, text))

def test_vector_search_project_filter_reaches_beyond_topk():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t.db"))
        base = [0.0] * 384
        near = list(base); near[0] = 1.0          # query direction
        mid = list(base); mid[0] = 0.9
        far = list(base); far[0] = 0.1            # project B, farthest
        _insert_chunk(conn, 1, "s1", "projA", "claude-code", "a one", near)
        _insert_chunk(conn, 2, "s1", "projA", "claude-code", "a two", mid)
        _insert_chunk(conn, 3, "s2", "projB", "codex", "b one", far)
        conn.commit()

        spy = _SpyModel()
        spy.embed = lambda texts, **kw: [np.array(near, dtype=np.float32)]

        results = _vector_search(conn, spy, "q", k=2, project="projB")
        assert [r["id"] for r in results] == [3], "Filter must apply inside KNN, not after"
        results = _vector_search(conn, spy, "q", k=2, source="codex")
        assert [r["id"] for r in results] == [3]
        conn.close()
```

- [ ] **Step 3: Прогнать — падает** (нет колонок metadata / нет параметров)

- [ ] **Step 4: Имплементация**

`src/deja/db.py` — DDL chunks_vec:

```python
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding float[384],
            source TEXT,
            project_path TEXT
        )
    """)
```

`src/deja/indexer.py` — в `_upsert_chunk` оба insert'а в chunks_vec:

```python
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec (rowid, embedding, source, project_path) VALUES (?, ?, ?, ?)",
            (chunk_id, vec_bytes, source, chunk["project_path"]),
        )
```
(во второй ветке — обычный `INSERT INTO chunks_vec (rowid, embedding, source, project_path) VALUES (?, ?, ?, ?)`)

`src/deja/search.py` — заменить `_vector_search`, `_fts_search`, `hybrid_search`; удалить `_annotate_source`:

```python
def _vector_search(conn, model, query: str, k: int = 20,
                   project: str = None, source: str = None) -> list[dict]:
    query_embedding = list(model.embed(["query: " + query]))[0]
    where = ["embedding MATCH ?", "k = ?"]
    params = [serialize_f32(list(query_embedding)), k]
    if source:
        where.append("source = ?")
        params.append(source)
    if project:
        where.append("project_path = ?")
        params.append(project)
    rows = conn.execute(
        f"""
        WITH vec_results AS (
            SELECT rowid, distance
            FROM chunks_vec
            WHERE {' AND '.join(where)}
            ORDER BY distance
        )
        SELECT c.id, c.session_id, c.message_index, c.timestamp,
               c.project_path, c.chunk_text, c.tool_result_text, c.source,
               v.distance
        FROM vec_results v
        JOIN chunks c ON c.id = v.rowid
        """,
        params,
    ).fetchall()
    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "source": r[7], "distance": r[8],
        }
        for r in rows
    ]

def _fts_search(conn, query: str, k: int = 20,
                project: str = None, source: str = None) -> list[dict]:
    escaped = fts5_escape(query)
    sql = """
        SELECT c.id, c.session_id, c.message_index, c.timestamp,
               c.project_path, c.chunk_text, c.tool_result_text, c.source,
               rank
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        WHERE chunks_fts MATCH ?
    """
    params = [escaped]
    if project:
        sql += " AND c.project_path = ?"
        params.append(project)
    if source:
        sql += " AND c.source = ?"
        params.append(source)
    sql += " ORDER BY rank LIMIT ?"
    params.append(k)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": r[0], "session_id": r[1], "message_index": r[2],
            "timestamp": r[3], "project_path": r[4], "chunk_text": r[5],
            "tool_result_text": r[6], "source": r[7], "fts_rank": r[8],
        }
        for r in rows
    ]

def hybrid_search(
    conn, model, query: str, limit: int = 10,
    project: str = None, date_from: str = None, date_to: str = None,
    source: str = None,
) -> list[dict]:
    k = 100 if (date_from or date_to) else 20
    vec_results = _vector_search(conn, model, query, k=k, project=project, source=source)
    fts_results = _fts_search(conn, query, k=k, project=project, source=source)
    merged = _rrf_merge(vec_results, fts_results)

    if date_from:
        merged = [r for r in merged if r.get("timestamp", "") >= date_from]
    if date_to:
        merged = [r for r in merged if r.get("timestamp", "") <= date_to]

    return merged[:limit]
```
Примечание: `serialize_f32(list(query_embedding))` — spy возвращает np.array, у реальной модели тоже есть `.tolist()`; `list()` работает для обоих. `time_decay` параметр удаляется здесь же, его смерть закрепляется в Task 14 (удаление функций).
ВНИМАНИЕ: `hybrid_search` больше не принимает `time_decay` — проверить вызовы в `cli.py`, `server.py`, `eval.py` (никто не передаёт — ок).

- [ ] **Step 5: Сьют зелёный.** Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
(существующий `test_search_with_project_filter` теперь проверяет SQL-фильтр — должен пройти)

- [ ] **Step 6: Commit**

```bash
git add src/deja/db.py src/deja/indexer.py src/deja/search.py tests/test_search.py
git commit -m "feat(search): filter project/source inside retrieval, not after

project and source are now vec0 metadata columns (KNN-level WHERE)
and SQL predicates in the FTS branch. Post-hoc filtering starved
filtered searches: chunks outside the global top-k were unreachable.
Removes _annotate_source — source ships in both SELECTs."
```

---

### Task 9: m4 — инклюзивный `date_to` + единая форма результатов

**Files:**
- Modify: `src/deja/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Падающие тесты**

```python
def test_date_to_includes_whole_day():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_indexed_db(tmp)
        results = hybrid_search(conn, model, "nginx", limit=5,
                                date_from="2026-03-30", date_to="2026-03-30")
        assert len(results) > 0, "date_to=YYYY-MM-DD must include that day"
        conn.close()

def test_result_dicts_have_uniform_keys():
    with tempfile.TemporaryDirectory() as tmp:
        conn, model = _make_indexed_db(tmp)
        for r in hybrid_search(conn, model, "nginx proxy", limit=5):
            assert "distance" in r and "fts_rank" in r and "source" in r
        conn.close()
```

- [ ] **Step 2: Прогнать — падают**

- [ ] **Step 3: Имплементация**

В `hybrid_search` блок date_to:

```python
    if date_to:
        if len(date_to) == 10:  # bare date — include the whole day
            merged = [r for r in merged if r.get("timestamp", "")[:10] <= date_to]
        else:
            merged = [r for r in merged if r.get("timestamp", "") <= date_to]
```

В `_rrf_merge`, перед `results.append(item)`:

```python
        item.setdefault("distance", None)
        item.setdefault("fts_rank", None)
```

- [ ] **Step 4: Сьют зелёный. Step 5: Commit**

```bash
git add src/deja/search.py tests/test_search.py
git commit -m "fix(search): date_to includes the whole day; uniform result keys"
```

---

### Task 10: M4 — gc не должен убивать сессию, живущую под другим путём

**Files:**
- Modify: `src/deja/indexer.py` (`gc_orphans`)
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Падающий тест**

```python
def test_gc_preserves_session_known_under_other_path():
    """Same session file moved/copied to another dir must survive gc of the old path (M4)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        model = get_embedding_model()
        lines = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}, "timestamp": "2026-01-01T00:00:00Z", "uuid": "1"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "world"}]}, "timestamp": "2026-01-01T00:00:01Z", "uuid": "2"},
        ]
        os.makedirs(os.path.join(tmp, "old")); os.makedirs(os.path.join(tmp, "new"))
        path_old = _make_session(os.path.join(tmp, "old"), "sess-42.jsonl", lines)
        index_file(conn, model, path_old, "projOld")
        path_new = _make_session(os.path.join(tmp, "new"), "sess-42.jsonl", lines)
        index_file(conn, model, path_new, "projNew")

        # old path disappeared from disk scan; new path is known
        gc_orphans(conn, {path_new})
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] >= 1, \
            "Chunks of a session still known under another path must survive"
        assert conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0] == 1
        conn.close()
```

- [ ] **Step 2: Прогнать — падает** (чанки удалены: gc по session_id)

- [ ] **Step 3: Фикс `gc_orphans`** — внутри цикла:

```python
    known_sessions = {
        os.path.splitext(os.path.basename(p))[0] for p in known_paths
    }
    for (path,) in indexed:
        if path not in known_paths:
            session_id = os.path.splitext(os.path.basename(path))[0]
            if session_id not in known_sessions:
                _delete_file_chunks(conn, session_id)
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (path,))
            print(f"[deja] gc: removed orphan {path}", file=sys.stderr)
```
(`known_sessions` вычислить один раз перед циклом)

- [ ] **Step 4: Сьют зелёный. Step 5: Commit**

```bash
git add src/deja/indexer.py tests/test_indexer.py
git commit -m "fix(indexer): gc keeps chunks of sessions still known under another path"
```

---

### Task 11: m8 + ленивый cwd + батчинг эмбеддингов + счётчик ошибок

**Files:**
- Modify: `src/deja/cli.py` (`cmd_index`)
- Modify: `src/deja/parsers/codex.py` (`discover` ленивый), `src/deja/parsers/claude_code.py` (+`resolve_project`), `src/deja/parsers/base.py` (протокол)
- Modify: `src/deja/indexer.py` (embed-батчинг, счётчик ошибок)
- Test: `tests/test_parser_codex.py`

- [ ] **Step 1: Падающий тест ленивого discover**

В `tests/test_parser_codex.py` (рядом с `test_discover_walks_date_tree_and_reads_cwd` — этот тест ПЕРЕПИСАТЬ под новый контракт):

```python
def test_discover_is_lazy_and_resolve_project_reads_cwd(tmp_path, monkeypatch):
    from deja.parsers import codex
    root = tmp_path / "sessions" / "2026" / "06" / "01"
    root.mkdir(parents=True)
    f = root / "rollout-1.jsonl"
    f.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "/my/project"}}) + "\n",
                 encoding="utf-8")
    monkeypatch.setattr(codex.config, "CODEX_SESSIONS_DIR", str(tmp_path / "sessions"))

    found = list(codex.discover())
    assert found == [(str(f), None)], "discover must not read files eagerly"
    assert codex.resolve_project(str(f)) == "/my/project"
```

- [ ] **Step 2: Прогнать — падает** (discover возвращает cwd)

- [ ] **Step 3: Имплементация**

`src/deja/parsers/codex.py`:

```python
def discover() -> Iterator[tuple[str, str]]:
    """Walk ~/.codex/sessions/YYYY/MM/DD/*.jsonl, yield (path, None).

    project_path (cwd from session_meta) is resolved lazily via
    resolve_project() — only for files that actually need indexing.
    We walk the tree directly rather than trusting session_index.jsonl,
    which can drift (per clean-my-agent maintainer's note on #8).
    """
    root = config.CODEX_SESSIONS_DIR
    if not os.path.isdir(root):
        return
    pattern = os.path.join(root, "**", "*.jsonl")
    for path in glob.iglob(pattern, recursive=True):
        if os.path.basename(path) == "session_index.jsonl":
            continue
        yield path, None


resolve_project = _read_session_cwd
```
(строку `resolve_project = ...` поставить ПОСЛЕ определения `_read_session_cwd`)

`src/deja/parsers/claude_code.py` — в конец файла:

```python
def resolve_project(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

`src/deja/parsers/base.py` — в Protocol добавить:

```python
    def resolve_project(self, path: str) -> str:
        """Resolve project_path for a discovered file (used when discover yields None)."""
        ...
```

`src/deja/cli.py` — `cmd_index`, заменить блок от загрузки модели до gc:

```python
        sources = all_sources() if args.source == "all" else [args.source]
        print(f"[deja] sources: {', '.join(sources)}", file=sys.stderr)

        files = _collect_files(sources)
        print(f"[deja] found {len(files)} JSONL files", file=sys.stderr)

        known_paths = {path for path, _, _ in files}
        work = [
            (path, project, src) for path, project, src in files
            if check_needs_reindex(conn, path) is not False
        ]

        if not work:
            print("[deja] nothing to index", file=sys.stderr)
        else:
            print(f"[deja] {len(work)} files need indexing, loading model...", file=sys.stderr)
            model = get_embedding_model()
            for i, (path, project, src) in enumerate(work):
                if project is None:
                    project = get_parser(src).resolve_project(path)
                print(
                    f"[deja] [{i+1}/{len(work)}] [{src}] {os.path.basename(path)}",
                    file=sys.stderr,
                )
                index_file(conn, model, path, project, source=src)

        gc_orphans(conn, known_paths, sources=sources)
```
Импорт в cli.py: `from deja.indexer import get_embedding_model, index_file, gc_orphans, check_needs_reindex`.

`src/deja/indexer.py` — embed-батчинг (заменить цикл по EMBED_BATCH_SIZE):

```python
        texts = ["passage: " + c["chunk_text"] for c in chunks]
        all_embeddings = list(model.embed(texts, batch_size=EMBED_BATCH_SIZE))
```
(удалить переменную цикла; `EMBED_BATCH_SIZE` остаётся как параметр fastembed). ВНИМАНИЕ: spy-модели в тестах принимают `**kwargs` — совместимо.

Счётчик ошибок — в `index_file`: перед циклом батчей `insert_errors = 0`; в `except`: `insert_errors += 1`; после цикла:

```python
    if insert_errors:
        print(f"[deja] WARNING: {insert_errors} chunks failed to insert in {os.path.basename(path)}",
              file=sys.stderr)
```

- [ ] **Step 4: Сьют зелёный.** (старый `test_discover_walks_date_tree_and_reads_cwd` заменён новым)

- [ ] **Step 5: Commit**

```bash
git add src/deja/cli.py src/deja/parsers/ src/deja/indexer.py tests/test_parser_codex.py
git commit -m "perf(index): skip model load when nothing changed; lazy codex cwd; single embed call

cmd_index now checks which files need work before loading the ONNX
model (cron runs with no changes cost ~0). codex discover no longer
opens every rollout file — cwd is resolved only for files being
indexed. Embedding moved to one model.embed call with batch_size.
Failed chunk inserts are now counted and reported."
```

---

### Task 12: m5 + m7 — open_db helper, busy_timeout, schema-check в CLI, lock для redact

**Files:**
- Modify: `src/deja/db.py` (+`open_db`, +`ensure_schema`, +`SchemaMismatchError`)
- Modify: `src/deja/cli.py` (cmd_stats/cmd_search/cmd_redact используют helper; redact под lock)
- Modify: `src/deja/eval.py` (open_db)
- Modify: `src/deja/server.py` (`_check_schema` через ensure_schema)
- Test: `tests/test_db.py`, `tests/test_cli.py`

- [ ] **Step 1: Падающие тесты**

`tests/test_db.py`:

```python
def test_ensure_schema_raises_on_mismatch():
    from deja.db import open_db, ensure_schema, SchemaMismatchError
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        conn = init_db(db_path)
        ensure_schema(conn)  # current version — no raise
        conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        conn.commit()
        with pytest.raises(SchemaMismatchError):
            ensure_schema(conn)
        conn.close()

def test_open_db_sets_busy_timeout():
    from deja.db import open_db
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        init_db(db_path).close()
        conn = open_db(db_path)
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        conn.close()
```

`tests/test_cli.py` — тест, вызывающий САМ `cmd_redact` (а не копию его логики):

```python
def test_cmd_redact_function_directly(monkeypatch, capsys):
    import deja.cli as cli
    from deja.db import init_db
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "index.db")
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO chunks (session_id, message_index, split_index, timestamp,
               project_path, chunk_text, tool_result_text)
               VALUES ('s', 0, 0, '', 'p', 'api_key = sk-proj-abcdef1234567890abcdef1234567890', '')""")
        cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO chunks_fts (rowid, chunk_text, tool_result_text) VALUES (?, ?, '')",
                     (cid, "api_key = sk-proj-abcdef1234567890abcdef1234567890"))
        conn.commit(); conn.close()

        monkeypatch.setattr("deja.config.get_index_path", lambda: db_path)
        monkeypatch.setattr("deja.cli.get_index_path", lambda: db_path, raising=False)
        monkeypatch.setattr("deja.cli.get_index_dir", lambda: tmp, raising=False)
        cli.cmd_redact()

        from deja.db import open_db
        conn = open_db(db_path)
        row = conn.execute("SELECT chunk_text FROM chunks").fetchone()
        assert "sk-proj" not in row[0]
        conn.close()
```
ВНИМАНИЕ: `cmd_redact` импортирует `get_index_path` локально или на уровне модуля — проверить фактический импорт в cli.py и monkeypatch-ить соответствующее имя. Если в `cmd_redact` путь берётся через `from deja.config import get_index_path` на уровне модуля cli — достаточно `monkeypatch.setattr("deja.cli.get_index_path", ...)`.

- [ ] **Step 2: Прогнать — падают** (нет open_db/ensure_schema)

- [ ] **Step 3: Имплементация**

`src/deja/db.py`:

```python
class SchemaMismatchError(RuntimeError):
    pass


def open_db(db_path: str, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def ensure_schema(conn: sqlite3.Connection):
    db_version = int(get_meta(conn).get("schema_version", "0"))
    if db_version != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"index schema v{db_version}, expected v{SCHEMA_VERSION} — run 'deja index'"
        )
```

`src/deja/cli.py` — `cmd_stats`, `cmd_search`, `cmd_redact`: заменить ручные блоки `sqlite3.connect + enable_load_extension + sqlite_vec.load` на:

```python
    from deja.db import open_db, ensure_schema, SchemaMismatchError
    conn = open_db(index_path)            # cmd_search: open_db(index_path, check_same_thread=False)
    try:
        ensure_schema(conn)
    except SchemaMismatchError as e:
        print(f"[deja] {e}", file=sys.stderr)
        sys.exit(1)
```
`cmd_redact` дополнительно обернуть в lock:

```python
def cmd_redact():
    lock_fd = _acquire_lock()
    try:
        ...весь текущий код...
    finally:
        _release_lock(lock_fd)
```

`src/deja/eval.py` — заменить ручное открытие на `open_db(index_path, check_same_thread=False)`.

`src/deja/server.py` — `_check_schema`:

```python
def _check_schema(conn):
    from deja.db import ensure_schema, SchemaMismatchError
    try:
        ensure_schema(conn)
    except SchemaMismatchError as e:
        raise ToolError(str(e))
```

- [ ] **Step 4: Сьют зелёный. Step 5: Commit**

```bash
git add src/deja/db.py src/deja/cli.py src/deja/eval.py src/deja/server.py tests/
git commit -m "refactor(db): open_db helper with busy_timeout + schema check in CLI

stats/search/redact/eval now share one connection helper (busy_timeout
5000 — no more 'database is locked' next to a running indexer), check
the schema version before querying, and redact takes the index lock."
```

---

### Task 13: m6 + m3 — id в get_session_chunks, понятная ошибка eval

**Files:**
- Modify: `src/deja/server.py` (`_do_get_session`)
- Modify: `src/deja/eval.py` (`evaluate`)
- Test: `tests/test_server.py`, `tests/test_cli.py`

- [ ] **Step 1: Падающие тесты**

`tests/test_server.py` — в `test_do_get_session` добавить assert:

```python
        assert "id" in result[0], "chunks must carry id so get_context can be chained"
```

`tests/test_cli.py`:

```python
def test_eval_missing_golden_exits_with_message(capsys):
    import pytest
    from deja.eval import evaluate
    from deja.db import init_db
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "index.db")
        init_db(db_path).close()
        with pytest.raises(SystemExit):
            evaluate(golden_path=os.path.join(tmp, "nope.json"), index_path=db_path)
        err = capsys.readouterr().err
        assert "golden" in err.lower()
```

- [ ] **Step 2: Прогнать — падают**

- [ ] **Step 3: Имплементация**

`src/deja/server.py` `_do_get_session`:

```python
    rows = conn.execute(
        """SELECT id, chunk_text, message_index, timestamp, project_path
        FROM chunks WHERE session_id = ? ORDER BY message_index, split_index""",
        (session_id,),
    ).fetchall()
    return [
        {"id": r[0], "chunk_text": r[1], "message_index": r[2],
         "timestamp": r[3], "project_path": r[4]}
        for r in rows
    ]
```

`src/deja/eval.py` — в начале `evaluate`, сразу после вычисления `golden_path`:

```python
    if not os.path.exists(golden_path):
        print(f"[deja] golden pairs file not found: {golden_path}", file=sys.stderr)
        print("[deja] pass --golden path/to/golden_pairs.json "
              "(JSON list of {query, expected_sessions})", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 4: Сьют зелёный. Step 5: Commit**

```bash
git add src/deja/server.py src/deja/eval.py tests/
git commit -m "fix(api): get_session_chunks returns chunk ids; eval reports missing golden file clearly"
```

---

### Task 14: Ниты — мёртвый код, abspath, is False, fts-orphans в stats

**Files:**
- Modify: `src/deja/parsers/claude_code.py` (удалить `get_file_end_offset`)
- Modify: `src/deja/parser.py` (убрать из re-export)
- Modify: `src/deja/search.py` (удалить `_apply_time_decay`, `TIME_DECAY_ALPHA`, импорты math/datetime)
- Modify: `src/deja/indexer.py` (`needs is False`)
- Modify: `src/deja/cli.py` (удалить `import glob`; fts-orphans в stats)
- Modify: `src/deja/config.py` (abspath)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Тест config**

```python
def test_relative_deja_index_path_resolves_absolute(monkeypatch):
    from deja import config
    monkeypatch.setenv("DEJA_INDEX_PATH", "index.db")
    d = config.get_index_dir()
    assert os.path.isabs(d), f"Expected absolute dir, got '{d}'"
```

- [ ] **Step 2: Прогнать — падает** (dirname('index.db') == '')

- [ ] **Step 3: Имплементация**

`src/deja/config.py`:

```python
def get_index_dir() -> str:
    env = os.environ.get("DEJA_INDEX_PATH")
    if env:
        return os.path.dirname(os.path.abspath(env))
```
(остальное без изменений)

`src/deja/parsers/claude_code.py`: удалить функцию `get_file_end_offset` (строка `parse = parse_jsonl_file` остаётся).
`src/deja/parser.py`: убрать `get_file_end_offset` из импорта и `__all__`.
`src/deja/search.py`: удалить `TIME_DECAY_ALPHA`, `_apply_time_decay`, `import math`, `from datetime import datetime, timezone` (если Task 8 ещё не убрал параметр — убрать здесь).
`src/deja/indexer.py`: `if needs == False:` → `if needs is False:`.
`src/deja/cli.py`: удалить `import glob`; в `cmd_stats` после vec-orphans добавить:

```python
    orphans_fts = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE rowid NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    if orphans_fts:
        issues.append(f"{orphans_fts} orphan fts rows")
```

- [ ] **Step 4: Сьют зелёный + grep-чистота**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
grep -rn "time_decay\|get_file_end_offset\|import glob" src/deja/ || echo CLEAN
```
Expected: passed + CLEAN.

- [ ] **Step 5: Commit**

```bash
git add src/deja/ tests/test_cli.py
git commit -m "chore: remove dead code (time_decay, get_file_end_offset), absolute DEJA_INDEX_PATH, fts orphan check in stats"
```

---

### Task 15: CI — права release workflow + порядок job'ов

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Правка**

Удалить top-level блок:

```yaml
permissions:
  contents: write
  id-token: write
```

В job `publish-pypi` добавить:

```yaml
  publish-pypi:
    runs-on: ubuntu-latest
    needs: [build]
    permissions:
      id-token: write
```

В job `github-release` добавить права и зависимость от публикации (release не создаётся, если PyPI-публикация упала):

```yaml
  github-release:
    runs-on: ubuntu-latest
    needs: [build, publish-pypi]
    permissions:
      contents: write
```

- [ ] **Step 2: Валидация синтаксиса**

```bash
.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml', encoding='utf-8')); print('YAML OK')"
```
(если pyyaml не установлен в venv: `pip install pyyaml` или проверить глазами отступы)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: scope id-token to publish job, create GitHub release only after PyPI publish"
```

---

### Task 16: Документация — CHANGELOG (0.3.1 + 0.4.0) и README

**Files:**
- Modify: `CHANGELOG.md`, `README.md`, `pyproject.toml` (version → 0.4.0)

- [ ] **Step 1: CHANGELOG — вставить ПЕРЕД секцией `## 0.3.0`:**

```markdown
## 0.4.0 (unreleased)

### Breaking Changes
- **Schema v2** — first `deja index` after upgrade rebuilds the index automatically (full reindex, ~30-40 min for a large index). Old v1 indexes are migrated in place.

### Features
- **Multi-source indexing** — parser registry; Codex CLI sessions (`~/.codex/sessions/`) indexed alongside Claude Code (#8 Phase 1). `--source` filter in CLI, MCP `search`, and library API
- **e5 prefixes** — embeddings now use `query:`/`passage:` prefixes the model was trained with; retrieval quality improves
- **Retrieval-level filters** — `project` and `source` filter inside vector KNN (vec0 metadata columns) and FTS, not after merging; filtered searches no longer starve
- **Secret redaction hardening** — redaction happens on full turn text before truncation and chunking (truncated private keys no longer leak); bare tokens caught: sk-ant/sk-proj, DigitalOcean, npm, PyPI, Figma, standalone JWT, Google API, Stripe live, Telegram bot

### Bug Fixes
- Schema version mismatch no longer crashes init or loops full reindexes — proper in-place migration
- `deja index --source X` no longer garbage-collects other sources' chunks
- Live sessions: the last (still growing) turn is re-indexed as it grows instead of being frozen at first sight; consecutive assistant entries are accumulated into the turn instead of dropped
- A file containing only a user message (assistant not yet replied) is no longer marked fully indexed — the first turn is picked up on the next run
- File growth during indexing is detected on the next run (pre-parse stat recorded)
- gc keeps chunks of sessions still known under another path
- `date_to=YYYY-MM-DD` includes the whole day; search results have uniform keys (`distance`, `fts_rank`, `source`)
- `get_session_chunks` returns chunk `id` (chains into `get_context`)
- CLI commands share a connection helper with busy_timeout (no more `database is locked` next to a running indexer) and check schema version; `deja redact` takes the index lock
- `deja eval` reports a clear error when the golden file is missing

### Performance
- `deja index` skips loading the embedding model when nothing changed
- Codex discovery no longer opens every rollout file on every run
- Embedding batched in a single `model.embed` call

### CI
- `id-token` scoped to the PyPI publish job; GitHub release is created only after a successful PyPI publish

## 0.3.1 (2026-04-08)

- PyPI package renamed to `dejasearch` (#6)
- Release workflow: tag → CI → build → PyPI (trusted publishing) + GitHub Release
- Lazy-load embedding model — instant MCP server startup
```

- [ ] **Step 2: README правки**

1. Секция Performance — заменить строку `| RAM (indexing) | ~300 MB |` на `| RAM (indexing) | ~3 GB (ONNX model) |`.
2. Секция Install/Add to Claude Code — заменить пример конфига на pip-установку:

```json
"deja": {
    "type": "stdio",
    "command": "deja",
    "args": ["serve"],
    "env": { "PYTHONUNBUFFERED": "1" }
}
```
с примечанием: `deja` должен быть в PATH (`pip install dejasearch`); вариант с полным путём до `.venv/Scripts/deja.exe` оставить как альтернативу для source-установки.
3. В описание `--source`/multi-source добавить примечание о формате `project`: для claude-code — кодированное имя директории (`C--Users-Oleg--projects-x`), для codex — рабочая директория сессии как есть (`/home/user/project`).
4. В раздел eval (если упоминается) — golden-файл приватный, в репо не входит; формат: JSON list of `{"query": ..., "expected_sessions": [...]}`.

- [ ] **Step 3: pyproject version**

```toml
version = "0.4.0"
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md pyproject.toml
git commit -m "docs: changelog for 0.3.1 and 0.4.0, fix RAM figure, pip-first MCP config, source formats"
```

---

### Task 17: E2E-верификация, реиндекс, сравнение eval, релиз

**ЧЕКПОИНТ: шаги 4+ требуют явного ок Олега (push, merge, tag).**

- [ ] **Step 1: Полный сьют + ручной smoke**

```bash
cd /c/Projects/deja && .venv/Scripts/python.exe -m pytest tests/ -q
```
Expected: все passed (~60+ тестов).

- [ ] **Step 2: Реальный реиндекс локального индекса** (бэкап сделан в Task 0; миграция v1→v2 отработает автоматически; фоновый запуск, ~30-40 мин)

```bash
cd /c/Projects/deja && .venv/Scripts/python.exe -m deja.cli index 2>&1 | tail -5
.venv/Scripts/python.exe -m deja.cli stats
```
Expected: stats показывает Health: OK, schema v2, число чанков сопоставимо с прежним (27-28k+; будет больше за счёт m1-аккумуляции и codex-сессий).

- [ ] **Step 3: Eval против baseline**

```bash
.venv/Scripts/python.exe -m deja.cli eval --golden tests/golden_pairs.json 2>&1 | tee docs/plans/eval-after.txt
diff <(tail -6 docs/plans/eval-baseline.txt) <(tail -6 docs/plans/eval-after.txt) || true
```
Expected: MRR@5 ≥ baseline. Если MRR упал >10% — СТОП, разбираться (наиболее вероятная причина — префиксы e5; проверить, что обе стороны префиксованы).

- [ ] **Step 4 (ок Олега): push + PR**

```bash
git push origin feat/codex-parser
```
Обновить описание PR #26: блокеры B1-B3 закрыты, объём вырос (перечислить коммиты), v0.4.0 готов.

- [ ] **Step 5 (ок Олега): merge PR #26, tag**

```bash
git switch main && git pull && git merge --no-ff feat/codex-parser && git push
git tag v0.4.0 && git push origin v0.4.0
```
CI сам: tests → build → PyPI → GitHub Release.

- [ ] **Step 6: Верификация релиза**

```bash
pip index versions dejasearch
```
Expected: 0.4.0 на PyPI; GitHub Release с нотами из CHANGELOG (awk-экстракция найдёт `## 0.4.0` — перед тегом убедиться, что заголовок имеет вид `## 0.4.0 (YYYY-MM-DD)` с датой релиза, не "unreleased").

- [ ] **Step 7: Память**

Обновить `memory/projects/deja.md`: v0.4.0 released, все findings ревью 2026-06-13 закрыты (кроме явного out-of-scope), PR #26 смержен. Обновить строку в MEMORY.md.

---

## Сводка покрытия findings → tasks

| Finding | Task |
|---------|------|
| C1 потеря первого turn | 1 |
| B3 EOF-freeze codex | 2 |
| m1 consecutive assistants | 3 |
| m2 race роста файла | 4 |
| M1 redact до усечения/чанкинга | 5 |
| M2 голые токены | 6 |
| M6 e5-префиксы | 7 |
| M5 фильтры в retrieval (+_annotate_source, +source post-hoc) | 8 |
| m4 date_to, форма результатов | 9 |
| M4 gc session-collision | 10 |
| m8 model load, lazy cwd, embed batch, error counter | 11 |
| m5 busy_timeout/lock, m7 schema check | 12 |
| m6 ids, m3 eval error | 13 |
| ниты: dead code, abspath, is False, fts orphans | 14 |
| CI permissions/ordering | 15 |
| CHANGELOG 0.3.1+0.4.0, README RAM/config/formats | 16 |
| E2E, reindex, eval compare, release v0.4.0 | 17 |
| M3 schema_version | ✅ закрыт ранее (B1, `48cac25`) |
| B1, B2 | ✅ закрыты ранее (`48cac25`, `c45d866`) |
