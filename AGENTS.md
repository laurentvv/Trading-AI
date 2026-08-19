# AGENTS.md — Operating Notes for AI Assistants

This file gives future AI agents (Kilo, Codex, Copilot, etc.) the minimum context needed to work safely in this repo. Read it once before editing production code.

**Project in one paragraph** — Python 3.12 trading-decision pipeline. Multi-model ensemble (Scikit-Learn, TimesFM 2.5, TensorTrade PPO, NexusAI-Client unified cloud LLMs for text + vision + oil fundamentals + web research, sentiment, Vincent Ganne model) orchestrated by `main.py`. Outputs to `trading_journal.csv`, with optional Trading 212 demo/live execution via `--t212`. Virtualenv managed by `uv`. See `README.md` for the full feature list.

---

## 1. Deterministic Memory: The Four-File Discipline

Never rely solely on your context window to track project progress — it degrades, compresses, and is wiped. The **single source of truth** for system state lives in **four files on disk**, in `memory-bank/`. At every initialization, crash, or restart, read them to rebuild your state deterministically.

### 1.1 The four files (strict formats)

| File | Role | Lifecycle |
|---|---|---|
| **`memory-bank/feature_list.json`** | Complete structured map of all features (`id`, `name`, `description`, `status: pending\|in_progress\|completed`, `dependencies`). | Generated at initial planning; updated whenever a feature changes status. |
| **`memory-bank/contract.md`** | The technical validation contract — a list of strict, testable assertions (15–30 criteria). | **Frozen** just before the first line of code is written; the generator may not modify it afterwards. |
| **`memory-bank/progress.md`** | Macroscopic dashboard of the **current sprint** (objective + iteration milestones). Lets the agent instantly know what it is doing on restart. | Updated at the end of each loop iteration. |
| **`memory-bank/log.md`** | Chronological execution journal, **append-only**. Nothing is ever erased; each major event is stacked. | One entry added at the start and end of every action. |

> `log.md` entry format (mandatory): `## [YYYY-MM-DD] <phase> | <description>` where `<phase>` ∈ `init`, `gen`, `eval`, `fix`, …
> *(Historical product corrections live in `memory-bank/changelog.md` — kept separate from the sprint dashboard.)*

### 1.2 Operational directives for the execution loop

1. **Bootstrap phase** — Before any action, check these four files exist. If missing, create them per the formats above (see `memory-bank/contract.md` and `feature_list.json` for the live templates). If present, read them to rebuild immediate memory.
2. **Action phase** — Before executing a task, write the corresponding line to `log.md`.
3. **Sync phase** — After every file write or test run, update the matching status file (`progress.md` or `feature_list.json`).
4. **Error handling** — If an exception occurs or the process is interrupted, the valid state is the one extracted from the **last line of `log.md`** combined with the assertions of `progress.md`.

---

## 2. Non-obvious safety invariants — DO NOT BREAK

### 2.1 NexusAI-Client Unified Cloud Architecture (Zero Local LLM)

All local LLMs (Ollama, local GGUF models) have been completely removed and replaced by **[`NexusAI-Client`](https://github.com/laurentvv/NexusAI-Client)** (`nexusai-client>=0.3.1`):

- **Unified Gateway**: All text and vision calls route through `AIGateway` (`AIGateway.auto_fallback()` and `AIGateway.auto_fallback_vision()`).
- **Resilience**: Zero-cost automatic fallback across configured cloud providers (Gemini Free/Pro, Groq, Cerebras, Mistral, Cohere, Nvidia NIM, OpenRouter, OrcaRouter, DeepSeek).
- **Dual-Layer JSON Defence**: Handled cleanly with `json_mode=True` and strict extraction (`_find_dict_with_keys`) validating required keys (`signal`, `confidence`, `analysis`).
- **Call Sites**:
  - `src/llm_client.py` (`get_llm_decision`, `get_visual_llm_decision`, `_query_nexus`, `_query_nexus_vision`)
  - `src/oil_bench_model.py` (`OilBenchModel._query_llm`)
  - `src/web_researcher.py` (`generate_search_query`)
  - `src/council/weekend_council.py` (`ask_llm`, `run_council`)
  - `morning_brief/morning_brief.py`
  - `src/agents/solver.py` & `src/agents/annotator.py` (FinAcumen)

### 2.2 Other invariants (non-exhaustive)

- **T212 demo vs live** is governed by `T212_ENV` in `.env.t212` (demo is rate-limit-tolerant; live is not). Never commit credentials.
- **Order safety (GO-gate 1, 2026-08-19)**: order POSTs go through `post_order_market` (timeout 15 s; on network error the broker position is re-checked BEFORE any retry — never blind-retry a market order, the endpoint is not idempotent). `safe_request` is for read-only calls.
- **Broker-side protection (GO-gate 2)**: every open position must have a dedicated GTC stop order (`stop_order_id`/`stop_price` in the state); the ratchet moves it UP only (peak×0.90, cancel-and-replace). A position knowingly without a stop is a CRITICAL event. Market BUYs carry an attached `takeProfit` (+8 %) with a bare-order fallback if refused.
- **Fill confirmation (GO-gate 3)**: state and DB are written only after the fill is observed at the broker (`averagePricePaid`), never on a bare 2xx.
- **Volatility is DAILY** (GO-gate 4): `compute_daily_volatility` (20-day std, never annualized) feeds the decision engine and weight manager — their thresholds are daily-scale. Decision thresholds: buy 0.15 / sell −0.125 / strong ±0.4375/−0.5625 (rescaled ×1/0.8, behaviour preserved).
- **Data safety (GO-gate 5)**: synthetic macro data is FORBIDDEN (the old "Method 4" is deleted); macro caches TTL 7 days; the price-cache fallback refuses caches older than 3 days (no trading on stale data).
- **Scheduler (GO-gate 6)**: single instance enforced by `scheduler.lock` (O_EXCL + PID + stale 2 h, kept fresh by a lock-keeper thread); the loop survives any non-KeyboardInterrupt exception; morning brief has catch-up (runs any time after 01:00 when missing that day).
- **Per-ticker budget**: `INITIAL_BUDGETS` dict (default 1000€ per ticker), **not** the historical 5000€ hardcoded fallback.
- **DB write isolation (`write_db`)**: `EnhancedTradingSystem(write_db=...)` controls whether the simulation step (`_execute_hypothetical_trade`) writes to `trading_history.db`. In T212 mode, `main.py` passes `write_db=not is_t212` → **only `t212_executor` writes to the DB, after a broker-confirmed fill**. The simulation still runs (for internal reporting) but does not pollute the DB with phantom trades. **Never set `write_db=True` in `--t212` mode**.
- **T212 quantity precision**: governed by `TICKER_QUANTITY_PRECISION` dict in `src/t212_executor.py`. Fallback is `DEFAULT_QUANTITY_PRECISION = 2`. Price fields use `PRICE_DECIMALS = 2`.
- **Equity (GO-gate 7)**: per-ticker equity = `initial_budget + realized_pl (FIFO) + unrealized` — persisted as `state["equity"]`, written to the journal column `T212_Equity` and fed to `performance_monitor`. `current_capital` keeps its old sizing semantics (position value when open).
- **win_rate sentinel**: `_calculate_win_rate` (`src/performance_monitor.py`) returns `-1.0` (not `0.0`) when no closed trades exist or on error.
- **Cache staleness**: 1 day (`src/data.py`) — Parquet files older than that are auto-refreshed.
- **Cycle timeout**: 40 min (`CYCLE_TIMEOUT_SECONDS` in `main.py`).

---

## 3. Testing & validation commands

PowerShell note: `uv run pytest ...` may fail with "Failed to canonicalize script path" on some Windows shells — prefer `.venv\Scripts\python.exe -m pytest tests/` or `uv run pytest --basetemp=data_cache/test_tmp tests/`.

| Goal | Command |
|---|---|
| Mocked unit tests | `.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py tests/test_weekend_council.py tests/test_morning_brief_init.py -v` |
| Full mocked suite (GO-gates included) | `.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp` (ignores live harnesses: `test_crawl4ai`, `check_*`, `bench_*`, `run_short_backtest`) |
| Order-safety / stops / equity unit tests | `.venv\Scripts\python.exe -m pytest tests/test_t212_orders.py tests/test_scheduler_lock.py tests/test_equity_tracking.py tests/test_data_safety.py -v` |
| Live broker-stop probe (DEMO only, consent required) | `uv run python tests/check_t212_stops.py` |
| Live weekend council (NexusAI Cloud multi-provider) | `uv run python -m src.council.weekend_council --days 7` |
| Live LLM JSON harness (NexusAI Cloud) | `uv run python tests/check_llm_json.py` |
| Full pipeline, simulation | `uv run main.py --simul` |
| Full pipeline, T212 demo | `uv run main.py --t212` |
| Supervised scheduler (lock + auto-restart) | `.\start_scheduler.bat` |
| Morning Market Brief generation | `uv run python morning_brief/morning_brief.py` |

---

## 4. Where things live

- `main.py` — entry point (`--t212` / `--simul` / `--ticker`).
- `src/llm_client.py` — schemas, `_query_nexus`, `get_llm_decision` / `get_visual_llm_decision`, context injectors (`get_morning_brief_context`, `get_council_verdict_context`).
- `src/oil_bench_model.py` — `OilBenchModel._query_llm`.
- `src/web_researcher.py` — `generate_search_query` & `get_web_research_context_async`.
- `src/council/` — **Weekend Council**: async multi-persona LLM retrospective (`run_council`) across distinct cloud providers (Groq, Cerebras, Mistral, Cohere, OpenRouter/OrcaRouter/Nvidia, Gemini Free & Pro), writes `docs/council_reports/council_report_YYYY-MM-DD.md`.
- `morning_brief/` — **Morning Market Brief**: overnight analytical synthesis powered by `NexusAI-Client`.
- `src/agents/` — **FinAcumen**: cognitive agent for deep market reasoning.
- `tests/check_llm_json.py` — live JSON diagnostic harness for NexusAI providers.
- **`memory-bank/`** — the deterministic 4-file state (§1).

---

## 5. Branching, Commits & Environment Rules

- **OS Constraint (CRITICAL)**: DEV and PROD both run on **Windows**. Never suggest Linux-only commands (`rm -rf`, `ls`, `cat`) — use PowerShell (`Remove-Item -Recurse -Force`, `Get-ChildItem`) or CMD (`rmdir /s /q`).
- **Cache Invalidation**: `data_cache/` and `logs_prod/` are gitignored.
- **Never commit secrets** (`.env*`, `*.db`, `data_cache/`, `logs_prod/` are gitignored).
- **Do not push** without explicit user request.
