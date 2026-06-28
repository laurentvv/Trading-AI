# ADR-003 — Weekend Council: Multi-Model Retrospective as the 11th Consensus Voice

**Status**: Accepted
**Date**: 2026-06-28
**Branch**: `feat/weekend-council-10527370498881876130`
**Decision owner**: User (explicit request) — design & implementation by ZCode
**Inspiration**: [`0xNyk/council-of-high-intelligence`](https://github.com/0xNyk/council-of-high-intelligence)
**Related**: `AGENTS.md` §4 (`src/council/`), `COUNCIL_TEST_PROD.md`

---

## 1. Context

The per-cycle decision engine (`enhanced_decision_engine.py`) aggregates 10
models into a weighted consensus, but each model sees **only the current
instant**. There was no mechanism for a *strategic retrospective* — no voice
that connected the dots across a whole week: "the models voted BUY 412 times
but accuracy is 0%", "CRUDP.PA bled -18.8% on 13 CRITICAL alerts", "the bullish
bias is structural, not situational".

The weekend council fills that gap. Adapted from
[`0xNyk/council-of-high-intelligence`](https://github.com/0xNyk/council-of-high-intelligence)
(an 18-persona Claude skill), it is a weekly, async, multi-persona LLM
deliberation that runs on Saturday at 01:00 and feeds its verdict back into the
real-time consensus as the 11th weighted voice.

---

## 2. The Decision — three integration levels

The council was built across three escalating levels of impact:

| Level | Mechanism | Impact |
|---|---|---|
| **1. Standalone report** | Saves `docs/council_reports/council_report_YYYY-MM-DD.md` | Human-readable retrospective |
| **2. Soft context** | Verdict injected into `construct_llm_prompt` (via `get_council_verdict_context`) | Hopes the LLM weighs it |
| **3. Coded vote** | Parsed stance becomes a `ModelDecision(model_name="council")` at weight 0.10 | Measurable: 9.5% of consensus |

**Level 3 is the load-bearing integration.** Levels 1 and 2 are preserved
(redundant context + human artefact), but only Level 3 guarantees the council
actually moves the needle — and only after the code-review fixes (see §6).

---

## 3. Architecture

### 3.1 Six personas on five model families

The core design principle of the original council: **genuine reasoning diversity
comes from different model lineages, not from prompt costume changes on one
model.** Each persona is bound to a model chosen for role-affinity.

| Persona | Model | Family | Lane |
|---|---|---|---|
| Le Stratège | Gemma 4 12B (Q6_K) | Google | Macro long-term |
| Le Gestionnaire de Risque | GLM-4.6V-Flash (Q6_K) | Zhipu/Z.ai | Infrastructure / capital |
| Le Quant | Qwen 3.5 9B | Alibaba | Statistical rigour |
| Le Sceptique | LFM 2.5 (Mamba, 1.2B) | Liquid AI | System bias |
| Le Tacticien | Mistral Nemo 12B (Q6_K) | Mistral | Short-term execution |
| Le Comportementaliste | LFM 2.5 (Mamba, 1.2B) | Liquid AI | Market psychology |
| **Le Juge** | **Qwen3.5-9B-MTP (Q6_K)** | Alibaba | Structured synthesis |

LFM is reused (Sceptique + Comportementaliste) but their **targeted questions**
keep them in distinct lanes (system-bias vs market-bias). The Judge gets the
strongest reasoning model (IFEval 91.5, 262K context) because its job —
synthesising a 6-voice transcript into a structured verdict — is the hardest.

### 3.2 Four-round protocol + anti-groupthink

```
Round 0 — Problem Restate Gate
  Each model reframes "what is the real question this week?" in one sentence.
  Divergent reformulations flag a poorly-framed problem.

Round 1 — Independent Analysis (targeted questions)
  Each model answers a persona-specific question (not a generic "analyse").
  Must end with: STANCE: BUY|SELL|HOLD (confiance: XX%)
  ┌─ Dissent quota: if ≥2/3 majority converges on one stance,
  │  the most confident member is FORCED to steelman the opposite.
  └─ (anti-groupthink from the original design)

Round 2 — Directed Debate (1-vs-1)
  Each member critiques ONE assigned opponent (Stratège↔Sceptique,
  Risk↔Quant, Tacticien↔Comportementaliste). Not a free-for-all.

Round 3 — Judge Verdict (Unresolved-first)
  Structure imperative:
    0. Ce que le conseil NE PEUT PAS déterminer (2-3 uncertainties)
    1. Décompte des positions (weighted tally)
    2. Désaccords clés + Judge's position
    3. Forces/faiblesses
    4. Leçons
    5. Recommandations
    + VERDICT_TICKER block (parsed by main.py)
```

### 3.3 Context — real PROD data

The council does **not** analyse a generic market feed. It ingests the project's
own operational data, which is what makes the retrospective valuable:

| Section | Source | What it exposes |
|---|---|---|
| Transactions | `trading_history.db` | Recent executed trades |
| Portfolio | `trading_history.db` | Position states |
| Model signals | `trading_history.db` | Per-cycle ensemble votes |
| **Model accuracy** | `model_performance.db` | signal_predicted vs actual_outcome — *did the models get it right?* |
| **Metrics & alerts** | `performance_monitor.db` | Sharpe, drawdown, CRITICAL alerts |
| **Trading journal** | `trading_journal.csv` | BUY/SELL/HOLD distribution (bias detector) |
| Logs | `logs_prod/*.md` | Audit report + morning brief excerpts |

Each fetch is wrapped in try/except → section skipped cleanly if the source is
absent (graceful degradation, same pattern as the morning brief).

### 3.4 The Level 3 vote (load-bearing)

The Judge ends its verdict with a parseable block:

```
VERDICT_TICKER:
SXRV.DE: BUY (0.65)
CRUDP.PA: SELL (0.90)
```

At each cycle, `get_council_ticker_stance(ticker)` (in `llm_client.py`):
1. Loads the freshest report (`_load_fresh_council_report`, single source of freshness).
2. Isolates the **last** `VERDICT_TICKER` block (robust to prose mentions).
3. Parses the stance for the requested **trading ticker** (`SXRV.DE`, not `^NDX`).
4. Applies **linear age decay**: full confidence at day 0 → 0 at day 7.
5. Returns `(signal, confidence)` or `(None, 0.0)` (graceful skip).

The engine adds a `ModelDecision(model_name="council")` at weight 0.10 (9.5%
of consensus after renormalisation). The council is exempt from the adaptive
weight loop (`fixed_weight_models` in `adaptive_weight_manager.py`) — its
temporal relevance is handled by age-decay upstream, not by trade outcomes it
can't be measured against.

---

## 4. Invariants & coexistence

| Invariant | How it's respected |
|---|---|
| **ADR-002 weights** | `council: 0.10` is *added*; all other weights unchanged. Renormalised to 1.0 by the existing mechanism. |
| **Dual-layer JSON defence (§2.1)** | Not applicable — the council runs in prose mode (no JSON schema). The `VERDICT_TICKER` stance is parsed Python-side, not by the LLM. Think-channel debris is scrubbed by `strip_thinking_debris()` at source. |
| **Adaptive weight loop** | Council is in `fixed_weight_models` — never rescaled by performance scores (it has no resolvable outcome). |
| **Outcome DB tracking** | Council is excluded from `record_model_prediction` — its "correctness" can't be measured against per-cycle market direction. |
| **Cycle timeout / orphan-thread safety** | Council runs in an isolated subprocess (`schedule.py:run_weekend_council`), 1h timeout, Saturday 01:00 — cannot interfere with the trading loop. |

---

## 5. Consequences

**Positive:**
- The consensus now has a *memory*. A weekly SELL verdict on a bleeding ticker
  pulls the score down measurably for the whole following week (test:
  `test_council_sell_lowers_score_vs_no_council`).
- The multi-model design produces genuine analytical divergence (validated
  live: Gemma macro vs GLM infrastructure vs Qwen math vs LFM bias).
- Age decay means a stale verdict fades to zero exactly when it should — no
  week-old strategic call overriding fresh real-time signals.

**Negative / accepted costs:**
- ~13-20 LLM calls per week on CPU. With the generous token budgets tuned for
  thinking models (`num_predict=8192` members / `12000` Judge, `num_ctx=32768`
  members / `65536` Judge), a full run can take several hours on a CPU-only
  box. The scheduler allows a **48-hour window** (`COUNCIL_TIMEOUT = 172800`)
  and a 15-minute per-call Ollama timeout (`_OLLAMA_TIMEOUT = 900`) so the
  council runs safely over the weekend without being killed mid-deliberation.
- The Judge must emit the `VERDICT_TICKER` block reliably. If it doesn't (LLM
  non-compliance in prose mode), the vote is *gracefully skipped* (not crashed),
  but the Level 3 impact is null until the prompt is adjusted. This is the
  known fragility — the prompt is hardened but not schema-enforced.
- Requires 6 Ollama models (~40 GB) on PROD. `setup_council_models.py` handles
  this; missing models silently fall back to Gemma (reduced diversity, logged).

---

## 6. Code-review fixes (post-implementation)

A thorough audit of the initial Level 3 commit found the council vote was
**inert in production**. The fixes (commit `2ccd26c`):

1. **Ticker mismatch (CRITICAL)** — passed `self.analysis_ticker` (`^NDX`) but
   the verdict uses trading tickers (`SXRV.DE`). Never matched → always skipped.
   Fixed: pass `self.ticker`.
2. **Freshness duplication** — text-injection and vote used separate copies of
   the staleness logic (could disagree + TOCTOU). Unified into
   `_load_fresh_council_report()`.
3. **Adaptive weight drift** — the manager could silently rescale the 0.10
   weight via a neutral performance score. Fixed: `fixed_weight_models` exemption.
4. **Parser robustness** — French decimal comma (`0,65`), percent rescaling
   (`85` → `0.85`), `rfind` for block isolation, ticker class allows `^`/`=`.
5. **DB pollution** — council excluded from outcome tracking.

---

## 7. Reversal procedure

To disable the council vote without removing the feature:

1. In `src/config_weights.py`, set `"council": 0.0` (or remove the key).
2. The `ModelDecision` is still created but contributes zero weight; effectively
   reverts to Level 2 (soft context only).

To fully remove: revert the 4 commits
(`946c4f3`, `e7bc5a4`, `2ccd26c`, `f25a15b`) and delete `src/council/`.
