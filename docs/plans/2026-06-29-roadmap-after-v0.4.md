# Deja roadmap — after v0.4.0 (2026-06-29)

5 открытых issues. План — что/когда/в каком порядке.

## Контекст

v0.4.0 (29.06) закрыла Phase 1 #8 (Codex) + indexer/secrets фиксы. Schema 1 → 3, 82 теста, на PyPI. Открытых issues — 5:

| # | Title | Schema bump? | Reindex? | New dep? | Effort |
|---|---|---|---|---|---|
| #25 | Analytics command | Yes (sessions table) | Optional backfill | No | 1 день |
| #24 | Git-state filtering | Yes (chunks.git_branch) | Yes | No | 1-2 дня |
| #13 | Parent-child chunking | Yes (chunks.parent_id) | Yes | No | 3-4 дня |
| #12 | Cross-encoder reranking | No | No | TextCrossEncoder | 2-3 дня |
| #8 Phase 2 | Cursor parser | No (uses source col) | No (extends data) | sqlite3 (stdlib) | 1-2 дня |
| #8 Phase 3 | Gemini parser | No | No | — | 1 день (если есть сэмплы) |
| #8 Phase 4 | OpenCode parser | — | — | — | DEPRIORITIZE — только metadata, semantic search бесполезен |

Суммарно — ~9-13 рабочих дней работы.

## Группировка по релизам

Принцип: **schema-bump'ы объединять в одну миграцию**, чтобы пользователь не делал reindex по 30-40 минут несколько раз.

### v0.5.0 — Quality bundle (schema-bump релиз)

Объединить три issue которые меняют storage:
- #25 Analytics → новая таблица `sessions` (pre-aggregate)
- #24 Git-state filter → колонки `chunks.git_branch` (+ опц. `git_head`, `git_dirty`)
- #13 Parent-child chunking → `chunks.parent_id` ИЛИ separate `parents`/`children`

Schema bump 3 → 4. Один reindex закрывает все три фичи.

**ETA:** 5-7 дней работы.

### v0.6.0 — Reranking

- #12 Cross-encoder reranking

Отдельный релиз потому что:
- Новый dependency (`TextCrossEncoder` из fastembed) — пользователи без него получают NoOp
- Меняет latency profile (~50ms → ~850ms warm)
- Должен быть configurable (default — спорно, см. open question)
- Не требует reindex

**ETA:** 2-3 дня.

### v0.7.0 — Cursor support

- #8 Phase 2 Cursor

Новый источник, не задевает существующее. Архитектура parser registry уже готова из v0.4.0 — просто новый модуль `parsers/cursor.py`.

**ETA:** 1-2 дня.

### v0.8.0 — Gemini support (условно)

- #8 Phase 3 Gemini

Зависит от живых сэмплов от пользователей или своих. Если нет — отложить до появления спроса.

**ETA:** 1 день (если сэмплы есть).

### Backlog / close as wontfix

- **#8 Phase 4 OpenCode** — `~/.local/share/opencode/storage/<bucket>/*.json` содержит только metadata, не транскрипты. Semantic search бесполезен. Предложить закрыть как `wontfix` с комментом «reopen when OpenCode persists full conversations».

---

## v0.5.0 — детальный план

### Что входит
1. #25 — Analytics (`sessions` таблица + CLI)
2. #24 — Git-state filtering (`chunks.git_branch` column + search filter)
3. #13 — Parent-child chunking (refactor chunker + search)

### Schema migration v3 → v4

```sql
-- #24
ALTER TABLE chunks ADD COLUMN git_branch TEXT;
ALTER TABLE chunks ADD COLUMN git_head TEXT;       -- optional
ALTER TABLE chunks ADD COLUMN git_dirty INTEGER;   -- 0/1/NULL
CREATE INDEX idx_chunks_branch ON chunks(git_branch);

-- #13
ALTER TABLE chunks ADD COLUMN parent_id INTEGER REFERENCES chunks(id);
CREATE INDEX idx_chunks_parent ON chunks(parent_id);

-- #25
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  project_path TEXT,
  started_at TEXT,
  ended_at TEXT,
  turn_count INTEGER NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_sessions_project ON sessions(project_path);
CREATE INDEX idx_sessions_started ON sessions(started_at);
```

`_migrate_if_needed` дропает + пересоздаёт — full reindex. ~30-40 мин.

### Реализация по подзадачам

#### Task A. #25 — Analytics (1 день)

1. Schema: `sessions` table (выше)
2. Parser: при yield turn возвращать также `usage` (tokens) если есть в исходном record. Claude Code: `message.usage.{input_tokens,output_tokens,cache_creation_input_tokens,cache_read_input_tokens}`. Codex: возможно `usage.*` поле.
3. Indexer: на каждый proceeded turn — `UPSERT sessions` с инкрементом счётчиков. Считать start = MIN(timestamp), end = MAX(timestamp).
4. CLI `deja analytics`:
   - default — все 5 отчётов
   - `--top-cost N` (top N по input+output+cache_creation)
   - `--top-length N` (top N по turn_count)
   - `--by-project` (распределение)
   - `--by-day` / `--since 30d` (ASCII sparkline по дням)
   - `--by-tool` (top 10 tool_use names — отдельная таблица или сканировать chunks?)
   - `--json` (machine-readable, нет ASCII)
5. Tool usage: либо отдельная таблица `tool_calls(session_id, tool_name, count)`, либо greppать chunk_text по `[Tool: <name>]` (медленнее, но текущий формат). **Предлагаю отдельную таблицу для скорости** — добавить в migration.

#### Task B. #24 — Git-state filtering (1-2 дня)

1. Schema: добавить колонки + индекс (выше)
2. Parser claude_code: extract `gitBranch` с каждого record (поле уже есть в JSONL Claude Code)
3. Parser codex: extract `session_meta.payload.git.branch` (если есть)
4. Parser cursor / gemini: NULL (на момент v0.5.0 их ещё нет, но контракт ясен)
5. Chunker: добавить git_branch в chunk dict
6. Indexer: писать в новые колонки на INSERT/UPDATE
7. Search:
   - `hybrid_search(..., git_branch="X")` — equals
   - `hybrid_search(..., git_branch_prefix="feature/")` — LIKE
   - Применять post-RRF (как `project`)
8. MCP search tool: новые параметры
9. CLI: `deja search --git-branch X` / `--git-branch-prefix feature/`

#### Task C. #13 — Parent-child chunking (3-4 дня)

Самая интрузивная часть. Решения по дизайну:

**Решение 1: parent_id vs separate tables**
- `parent_id` колонка проще (одна таблица, существующий код почти не меняется). Parent — это chunk сам по себе (с `parent_id IS NULL`), child — chunk с `parent_id` указывающим на parent.
- Separate tables строже семантически, но требуют двух таблиц.
- **Рекомендую parent_id колонку** — minimum disruption.

**Решение 2: что считать parent**
- Parent = тот же chunk что сейчас (user + assistant turn, до 1500 chars) — то есть существующее поведение становится parent
- Child = одно сообщение из turn (user отдельно, assistant отдельно), embed только child
- Search возвращает unique parents

**Решение 3: embed strategy**
- Embed только children — у них семантика чище
- Parents хранятся для return-context, не embed'ятся
- `chunks_vec` содержит только child embeddings
- `chunks_fts` — на child text (или на parent? — для FTS keyword пользы больше от parent, но дубликат). Default — на child.

**Реализация:**

1. Chunker — переписать:
   - Из turn (user+assistant) создать parent chunk (как сейчас)
   - Из turn создать 2 child chunk'а (user отдельно, assistant отдельно)
   - Если user/assistant больше 400 tokens — split дальше (рекурсивно)
   - Связать через `parent_id`
2. Indexer: вставлять parent сначала (без embedding), потом children с parent_id (с embedding)
3. Search:
   - Hybrid retrieval ищет ТОЛЬКО среди children (`WHERE parent_id IS NOT NULL` в SQL)
   - Для каждого hit — fetch parent
   - Dedup parents
   - Return parents
4. `get_context`: использовать parent_id для navigation. ±window становится «N parents вперёд/назад» в той же session.
5. Eval delta: baseline до изменения, eval после. Ожидаем рост MRR (более точные hits).

**Риск:** в 2-3x больше chunks при индексации, БД растёт, дольше initial reindex.

### Acceptance criteria v0.5.0

- [ ] 82 → ~110 тестов (примерно +28 на 3 фичи)
- [ ] Migration v3 → v4 авто-detect, чёткое log сообщение, full reindex
- [ ] `deja analytics` все 5 отчётов работают, JSON output валиден
- [ ] `deja search --git-branch X` фильтрует
- [ ] Eval delta — MRR должен **вырасти или остаться** после parent-child (если упал — разбираться)
- [ ] CHANGELOG обновлён
- [ ] README обновлён про analytics, git filter, parent-child concept

### Риски v0.5.0

1. **Parent-child может ухудшить retrieval** на коротких turn'ах где child === parent. Mitigation: eval до/после.
2. **Migration overhead** на больших индексах (130 MB сейчас, после parent-child ~250-300 MB). Тестить на реальном индексе перед релизом.
3. **`sessions` pre-aggregate vs on-fly** — если на лету (без таблицы) проще, но медленнее на больших индексах. Pre-aggregate лучше для UX но требует write-path в indexer.
4. **gitBranch может быть NULL** для старых сессий или несовместимых форматов. Фильтр должен ignore NULL.

---

## v0.6.0 — Reranking (детально)

### Что входит
- #12 — Cross-encoder reranking через `BAAI/bge-reranker-base`

### Реализация

1. Dependency: `fastembed.TextCrossEncoder` — уже доступен в fastembed. Никаких новых pip-зависимостей.
2. Lazy load: модель грузить только при первом вызове (как embedding model сейчас).
3. Pipeline:
   ```
   hybrid_search → top-20 (RRF) → rerank → top-N (default 5)
   ```
4. Config:
   - `reranker_enabled: bool = True` (по умолчанию ON?)
   - `reranker_model: str = "BAAI/bge-reranker-base"` (можно override)
   - `reranker_top_k: int = 20` (сколько RRF результатов отдать в reranker)
   - Env: `DEJA_RERANK=0` чтобы отключить runtime
5. MCP search tool: новый параметр `rerank: bool = True`
6. CLI: `deja search --no-rerank` для сравнения

### Acceptance criteria

- [ ] Rerank можно полностью отключить env-var + параметром
- [ ] Latency baseline + after — задокументированы (warm vs cold)
- [ ] Eval delta положительная (или нейтральная) на golden_pairs.json
- [ ] ~5 новых тестов
- [ ] README обновлён

### Риски

1. **Cold start модели** — первый search в новой сессии будет 5-10 секунд (загрузка ~110M reranker). Mitigation: lazy load + log при загрузке.
2. **RAM:** +300-500 MB на reranker модель. Documented в README.
3. **Latency:** 50 → 850 ms — приемлемо для MCP, но субъективно медленнее. Default = ON или OFF? **Open question** (см. ниже).
4. **Quality regression на коротких queries** — reranker может занижать relevance для коротких запросов где FTS даёт точный match. Eval покажет.

---

## v0.7.0 — Cursor support (детально)

### Что входит
- #8 Phase 2 Cursor SQLite parser

### Реализация

1. Discover: `<UserStorage>/workspaceStorage/<hash>/state.vscdb` + `workspace.json` (для identification project_path)
   - Windows: `%APPDATA%/Cursor/User/workspaceStorage/`
   - macOS: `~/Library/Application Support/Cursor/User/workspaceStorage/`
   - Linux: `~/.config/Cursor/User/workspaceStorage/`
   - Каждый hash directory = workspace
2. Parser:
   - Open `state.vscdb` via stdlib `sqlite3`
   - Query: `SELECT [key], value FROM ItemTable WHERE key LIKE 'composer.composerData%'`
   - Также `cursorDiskKV` если нужно (по референсам)
   - Values — nested JSON strings, парсить рекурсивно
3. Mapping в стандартный turn format:
   - user / assistant messages
   - tool calls / outputs
   - timestamps (если есть)
   - **Token usage** обычно недоступен локально (документировано в issue)
4. Регистрация в `parsers/registry.py`
5. Тесты: фикстурный state.vscdb с известными conversations

### Acceptance criteria

- [ ] `deja index --source cursor` индексирует minimum 1 фикстурную сессию
- [ ] Search возвращает Cursor chunks с `source=cursor`
- [ ] ~10 тестов
- [ ] README — добавить cursor в Supported sources таблицу

### Риски

1. **Cursor меняет формат БД** — нужны fixtures для разных версий. Mitigation: версия в metadata.
2. **Workspace identification** — `workspace.json` может отсутствовать → fallback на hash как project_path.
3. **Nested JSON depth** — может быть глубокая структура, нужны guard rails против infinite recursion.

---

## v0.8.0 — Gemini (условно)

### Что входит
- #8 Phase 3 Gemini parser

### Pre-requisite
- **Живые сэмплы сессий Gemini CLI.** Без них кодить вслепую — антипаттерн.
- Conflicting reports paths: `~/.gemini/tmp` (cass) vs `~/.gemini` / `~/.config/gemini` / `~/Library/Application Support/Gemini` (clean-my-agent).

### Если решаем делать:
1. Олег ставит Gemini CLI, делает 2-3 сессии разных типов
2. Реверс-инжиниринг формата по сэмплам
3. Реализация parser по той же архитектуре
4. Тесты с фикстурой

**ETA:** 1 день если сэмплы есть. **Иначе — отложить.**

---

## Phase 4 OpenCode — рекомендация close

Изучить ещё раз `~/.local/share/opencode/storage/`. Если действительно только metadata — закрыть issue с комментом:

> Per clean-my-agent docs and own inspection: OpenCode storage buckets contain only metadata (session ids, timestamps, paths), not full chat transcripts. Semantic search has no useful target. Reopen when OpenCode adds conversation persistence.

---

## Решения (приняты 2026-06-29)

1. **Порядок:** v0.5.0 первым (schema bump для 3 фич за один reindex выгоднее)
2. **#13 storage:** `parent_id` колонка в существующей chunks таблице (minimum disruption)
3. **#12 reranker default:** OFF, opt-in через `--rerank` flag / `DEJA_RERANK=1` env (cold start 5-10 сек заметен пользователю)
4. **#25 analytics:** pre-aggregate в `sessions` table (on-fly на 20k chunks медленно, complication индексера минимальная)
5. **#24 git_head / git_dirty:** скип на MVP, только `git_branch` (Claude Code JSONL содержит только branch; head/dirty потребуют отдельной логики читать `.git/HEAD` — плохой ROI сейчас)
6. **#8 Phase 4 OpenCode:** wontfix (подтверждено в issue body — storage содержит только metadata)
7. **Gemini:** отложить (нет сэмплов на руках)

---

## Последовательность работы (рекомендованная)

```
v0.5.0 (5-7 дней)
  Task A — Analytics (1 день)
  Task B — Git-state filter (1-2 дня)
  Task C — Parent-child chunking (3-4 дня)
  Migration testing + eval (полдня)
  Release
       │
       ▼
v0.6.0 (2-3 дня)
  Reranker integration
  Latency benchmark
  Eval comparison
  Release
       │
       ▼
v0.7.0 (1-2 дня)
  Cursor parser
  Fixtures + tests
  Release
       │
       ▼
v0.8.0 (1 день) — условно при наличии сэмплов
  Gemini parser
```

Промежуточные patch-релизы (v0.4.1, v0.5.1) — по факту багов после деплоя.

## Бюджет суммарно
- ~9-13 рабочих дней работы
- 4 минор-релиза
- 1 закрытие как wontfix
- Если делать по выходным/вечерам — растягивается на 1-2 месяца

## Зависимости от внешнего
- v0.5.0 — нет
- v0.6.0 — нет (fastembed уже в depends)
- v0.7.0 — нет (sqlite3 stdlib)
- v0.8.0 — **нужны Gemini сэмплы**
