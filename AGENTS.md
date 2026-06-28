# AGENTS.md — Operating Notes for AI Assistants

This file gives future AI agents (Kilo, Codex, Copilot, etc.) the minimum context needed to work safely in this repo. Read it once before editing production code.

## 1. Project in one paragraph

Python 3.12 trading-decision pipeline. Multi-model ensemble (Scikit-Learn, TimesFM 2.5, TensorTrade PPO, Gemma 4 12B LLM for text + vision + oil fundamentals + web research, sentiment, Vincent Ganne model) orchestrated by `main.py`. Outputs to `trading_journal.csv`, with optional Trading 212 demo/live execution via `--t212`. Virtualenv managed by `uv`. See `README.md` for the full feature list.

## 2. Non-obvious safety invariants — DO NOT BREAK

### 2.1 Dual-Layer JSON Defence (Gemma think mode)

Gemma 4 12B's `<|think|>` reasoning channel is **active** in production. JSON-extraction safety is provided by a **two-layer** defence — neither layer alone is sufficient:

- **Layer 1 (load-bearing)** — schema-strict `format: SCHEMA_*` parameter at every Ollama call site. Defined in `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`). Uses `additionalProperties: false`. Enforced server-side by Ollama.
- **Layer 2 (belt-and-braces)** — every system prompt ends with `"...never add a 'thought' key."`.

**Implications for code changes:**

- When editing any of the 4 call sites (`src/llm_client.py:188`, `src/llm_client.py:236`, `src/oil_bench_model.py:158`, `src/web_researcher.py:205`), **preserve both layers**. Do not strip the `"<|think|> "` prefix or the `"...never add a 'thought' key."` suffix.
- If you add a new LLM call site, **always** use a schema-strict `format: SCHEMA_*` value. Never use the loose `format:json` — `tests/check_llm_json.py` proves it leaks `<|channel>thought` debris even without `<|think|>`.
- If you change a schema, run `tests/check_llm_json.py` end-to-end before merging. Acceptance: all `*_schema*` and `oil_*` cases must be OK.

See `docs/ADR-001-think-mode-dual-layer-defence.md` for the full validation evidence and reversal procedure.

### 2.2 Other invariants (non-exhaustive)

- **T212 demo vs live** is governed by `T212_ENV` in `.env.t212`. Demo is rate-limit-tolerant; live is not. Never commit credentials.
- **Per-ticker budget**: `INITIAL_BUDGETS` dict (default 1000€ per ticker), not the historical 5000€ hardcoded fallback.
- **Cache staleness threshold**: 1 day (`src/data.py`). Parquet files older than that are auto-refreshed.
- **Cycle timeout**: 15 min (`CYCLE_TIMEOUT_SECONDS` in `main.py:39`). On timeout, `cancel_event` is set so the orphan thread cannot place a T212 order even if it finishes later.
- **Orphan-thread safety**: per-ticker `threading.Lock` serializes T212 order placement.

## 3. Testing & validation commands

| Goal | Command |
|---|---|
| Mocked unit tests (deterministic, no Ollama) | `.venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py tests/test_weekend_council.py -v` |
| Weekend council unit tests (mocked) | `.venv\Scripts\python.exe -m pytest tests/test_weekend_council.py -v` |
| Live weekend council (Gemini members + Ollama locals, ~5 min, 19 LLM calls) | `uv run python src/council/weekend_council.py --days 7` |
| Live LLM JSON harness (requires `ollama serve`) | `.venv\Scripts\python.exe tests/check_llm_json.py` |
| Full pipeline, T212 demo | `uv run main.py --t212` |
| Standalone prod backtest | `uv run backtest_prod.py` |
| Refresh parquet cache | `uv run refresh_cache.py` |

PowerShell note: `uv run pytest ...` may fail with "Failed to canonicalize script path" — prefer the direct `.venv\Scripts\python.exe -m pytest` form.

## 4. Where things live

- `main.py` — entry point, `--t212` / `--simul` / `--ticker` flags
- `src/llm_client.py` — schemas, `_query_ollama`, `get_llm_decision`, `get_visual_llm_decision`, `get_morning_brief_context` / `get_council_verdict_context` (context injectors into `construct_llm_prompt`)
- `src/oil_bench_model.py` — `OilBenchModel._query_llm` (call site #3)
- `src/web_researcher.py` — `generate_search_query` (call site #4)
- `src/enhanced_decision_engine.py` — consensus aggregator (note: does **not** currently consume the `failed: True` flag from `_fallback_decision` — silent degradation risk if `<|think|>` ever reintroduces JSON debris)
- `src/gemini_gateway.py` + `src/gemini_quota.py` — **two-tier Gemini cloud gateway** shared by `get_llm_decision` (JSON decisions), `get_visual_llm_decision` (vision), `web_researcher` (summaries) AND the Weekend Council (prose deliberations). Cascades (Pro → 3.5-flash → … → safety nets) + `QuotaTracker` (RPM/RPD + local `GEMINI_PAY_DAILY_CAP` billing cap, SQLite ledger). Returns `None` on exhaustion so callers fall back to free-llm / Ollama. See §8.
- `src/council/` — **Weekend Council**: async, multi-persona LLM retrospective (`run_council`). NOT a per-cycle consensus vote; runs Sat & Sun at 09:00 via `schedule.py`, writes `docs/council_reports/council_report_YYYY-MM-DD.md`. Its Judge verdict is injected into the decision prompt via `get_council_verdict_context()` (same pattern as the morning brief). **Core design** (per `0xNyk/council-of-high-intelligence`): 6 personas (Stratège / Risk Manager / Quant / Sceptique / Tacticien / Comportementaliste), each on a model chosen for role-affinity (`MEMBER_MODELS` in `council_prompts.py`: 5 families — Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / Mistral Nemo 12B / **Gemini 2.5 Flash**), with a 4-round protocol (Problem Restate Gate → Analysis with explicit STANCE → 1-vs-1 Debate → Judge synthesis) plus anti-groupthink mechanisms (dissent quota ≥2/3, unresolved-first verdict). The Judge runs on **Gemini 2.5 Pro** (paid cascade). **Hybrid cloud routing**: models prefixed `gemini:` go through the shared `GeminiGateway.deliberate()` (members → FREE cascade, Judge → PAID cascade) — same `QuotaTracker`/billing cap as the real-time decision calls; all other members run on local Ollama. On quota exhaustion / 429 / 503, the gateway returns `None` and the member falls back to local Ollama — the council never hard-fails. Models installed via `setup_council_models.py` (skips the `gemini:` ones — they need API keys, not a local pull). **Must run** `uv run python setup_council_models.py` on PROD after `git pull`, and set `GEMINI_API_KEY` / `GEMINI_API_KEY_PAY` in `.env` (see §8).
- `tests/check_llm_json.py` — live diagnostic harness, 10 cases × 3 schema families × 2 modes
- `docs/ADR-*.md` — architecture decision records
- `memory-bank/` — long-form context (techContext, activeContext, systemPatterns, progress, projectbrief, productContext, improvement_proposals)

## 5. Branching, Commits & Environment Rules

- **OS Constraint (CRITICAL)**: Both DEV and PROD environments run on **Windows**. Never suggest Linux-only commands like `rm -rf`, `ls`, or `cat`. Use PowerShell equivalents (`Remove-Item -Recurse -Force`, `Get-ChildItem`, etc.) or CMD (`rmdir /s /q`).
- **Cache Invalidation**: Folders like `data_cache/` and `logs_prod/` are gitignored. When pushing bug fixes related to corrupted data (e.g., TensorTrade policy collapse, broken EIA timestamps), you MUST explicitly remind the user to manually delete the affected caches on the PROD server after `git pull` (e.g., `Remove-Item -Recurse -Force data_cache\tensortrade` or running `uv run python refresh_cache.py`).
- **Council Models (CRITICAL for the weekend council feature)**: the council's diversity comes from each persona running on a **different** model lineage (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / Mistral Nemo 12B on local Ollama + Gemini 2.5 Flash/Pro on the cloud). When pushing the council feature, you MUST remind the user to (1) run `uv run python setup_council_models.py` on the PROD server after `git pull` — otherwise local members silently fall back to Gemma and the council degrades to "costume changes on one model"; AND (2) set `GEMINI_API_KEY` (free) and `GEMINI_API_KEY_PAY` (paid) in `.env` — otherwise the Sceptique, Comportementaliste and Judge fall back to local Ollama (slower, but still functional).
- Long-lived branch: `main` (production state, schema-defended).
- Validation branches: short-lived feature branches like `think-mode` for risky re-enablements.
- **Never commit secrets** (`.env*`, `*.db`, `data_cache/`, `logs_prod/` are gitignored — keep them that way).
- **Do not push** without explicit user request.

## 6. Known follow-ups (flagged, not in scope)

- `_THINKING_TOKENS` list is duplicated between `src/llm_client.py:52-56` and `tests/check_llm_json.py:388-392`. If extended, they must stay in sync — or be unified.
- `_fallback_decision`'s `failed: True` flag is not consumed by the consensus aggregator (`enhanced_decision_engine.py`). A failed LLM call is currently weighted as a plain HOLD vote — silent HOLD bias if extraction fails intermittently.
- `tests/check_llm_json.py` returns exit 1 if any case fails, including the documented-to-fail loose-format cases. The harness could be refactored to return 0 when only the expected-failure cases fail.
- **FinAcumen (June 2026) — repaired; converges; feeds the morning brief (which itself consumes `main.py` outputs).** `src/finacumen_main.py` now converges (`status: success`) after fixing `src/core/tools.py` (`lookup_ohlc` accepts a list of indicators → dict; rsi/sma/macd added; pd/np pre-injected so the sandbox's `__import__` ban is harmless) and `src/agents/solver.py` (prompt documents the real API; observation echoes fetched `data` when the LLM forgets to `print`; the execute-vs-final-answer branch keys on *non-empty* `python_code` / `action in BUY|SELL|HOLD`, not key presence).
  - **Coupling to the main project** is via shared data files, not a direct import: `main.py` writes `trading_journal.csv` (main.py:222), `trading.log`, and `performance_monitor.db`; `morning_brief/tools/analyze_trading_logs.py` + `morning_brief/tools/audit_portfolio_performance.py` read them; `schedule.py` appends the FinAcumen section into `morning_market_brief.md`. So FinAcumen **does** influence the daily decision brief.
  - **By design**, it is **not** an 11th vote in `enhanced_decision_engine.py` / `model_performance.db` (the real-time per-cycle consensus). Its role is strictly asynchronous and structural for the morning brief. *(The 11th-consensus-vote slot is now occupied by the Weekend Council — see §4 `src/council/`.)*
- `backtest_prod.py` reads `data_cache/` (repo root, ends ~2026-05-27) instead of `logs_prod/data_cache/` (prod, current). Its tables come back empty when the journal spans a period not covered by the root cache. Use `audit_prod_logs.py` for a backtest that reads the prod cache.

### 6.1 Resolved follow-ups (Late June 2026 - ADR-002 Suite & Architecture)

The end-of-June decision-model audit (`docs/ADR-002-decision-model-quality-audit.md`) remediated the structural bullish bias and the market-derived win_rate metric:

- **win_rate metric & Accuracy** — Replaced absolute 0 baseline. A `BUY` is now only rewarded if the return exceeds the `HOLD_NEUTRAL_RETURN_THRESHOLD` (0.5% buffer covering volatility/fees). `HOLD` is rewarded if the market is flat. This stops blindly optimistic models from farming points during slight market drift.
- **Bullish bias & Overfitting** — 
  - *TensorTrade*: Disabled continuous `model.learn(total_timesteps=500)` during cached inference, which was causing catastrophic policy collapse (stuck on BUY 1.00). 
  - *LLM Visual*: Increased temperature to 0.4 and hardened the prompt to force `HOLD` on ambiguous charts, ending deterministic output spam.
- **Hybrid LLM Architecture** — `get_llm_decision` now prioritizes a Cloud-based "Frontier Model" via `free-llm-api-keys` (for high-IQ text/macro analysis), with an instant, silent fallback to local Ollama (Gemma 4 12B) on API 503 errors. Vision (VL) strictly remains on Ollama, as free proxies reject image payloads.

Still open after ADR-002: isotonic tensortrade recalibration (cap is interim), upstream sentiment-data skew, unifying the two `regime_adjustments` dicts, `return_5d` population.

## 7. Tooling scripts (June 2026)

- `audit_prod_logs.py` — validates **all** files in `logs_prod/` (catalogue, SQLite integrity, parquet freshness, JSON/pkl, FinAcumen state) and runs a corrected backtest against `logs_prod/data_cache/`. Emits `logs_prod/audit_report.md`. Run with `uv run python audit_prod_logs.py`.
- `setup_council_models.py` — installs the **four local** Ollama model families the Weekend Council requires (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / Mistral Nemo 12B), and reports (without trying to pull) the cloud Gemini models. Each council persona runs on its **own model lineage** — this is the core of the design (genuine reasoning diversity, not costume changes). **Must be run on every PROD server** after `git pull` of the council feature, otherwise local members silently fall back to Gemma and the council loses its diversity. Run with `uv run python setup_council_models.py` (idempotent, skips installed models). Cloud Gemini members (Sceptique, Comportementaliste, Judge) need API keys in `.env`, not a local install — see §8.

## 8. Gemini cloud gateway (two-tier, shared)

The project has a **single** Gemini cloud layer — `src/gemini_gateway.py` (`GeminiGateway`) — shared by every cloud-routed caller: the real-time decision/vision calls (`get_llm_decision`, `get_visual_llm_decision`), the web summary, AND the Weekend Council. There is no second Gemini path; do not build one.

**Two keys, two tiers** (both in `.env`, gitignored — get them at https://aistudio.google.com/apikey):

| Env var | Tier | Notable models | Used by |
|---|---|---|---|
| `GEMINI_API_KEY` | **Free** | `gemini-2.5-flash` (Pro is 0/0/0 here) | Council members (Sceptique, Comportementaliste); web summary; fallback when paid tier exhausted |
| `GEMINI_API_KEY_PAY` | **Paid** | `gemini-2.5-pro`, `gemini-3.5-flash` | Council Judge (`gemini:pro`); real-time decision + vision |

**Routing convention**: a model string prefixed `gemini:` is routed to the gateway. In the council, the tier is inferred from the id — `"pro"` → paid cascade, otherwise free. The gateway internally runs a **cascade** (head model → safety nets) per tier and the `QuotaTracker` (`src/gemini_quota.py`) pre-flights each call against RPM/RPD caps plus a **local daily billing cap** (`GEMINI_PAY_DAILY_CAP`, default 200 paid calls/day). When the gateway can't serve a request it returns `None` (never raises) and the caller falls back to local Ollama.

**Public methods** (add new use-cases here, don't duplicate the SDK elsewhere):
- `decide(prompt)` → dict `{signal, confidence, analysis}` (JSON, paid-led) — used by `get_llm_decision`
- `analyze_chart(image_path)` → dict (vision, paid-led) — used by `get_visual_llm_decision`
- `summarize_web_context(text)` → str (free-only) — used by web research
- `deliberate(system, user, use_paid)` → str (prose, no JSON schema) — used by the Weekend Council

**Performance**: a Gemini call takes **0.4–30 s** vs **5–9 min** per call on CPU Ollama. Replacing the LFM 2.5 doublon + the local Judge with Gemini cut the council runtime from ~2h17 to a few minutes.
