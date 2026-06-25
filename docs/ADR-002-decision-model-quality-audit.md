# ADR-002 — Decision-Model Quality Audit & Bullish-Bias Remediation

**Status**: Accepted
**Date**: 2026-06-25
**Branch**: `fix/decision-model-quality-audit` → `main`
**Decision owner**: User (explicit request) — analysis & implementation by ZCode
**Related**: `docs/PLAN_DECISION_FIN_JUIN_2026.md` (go/no-go for DEMO→REAL T212)

---

## 1. Context

The end-of-June go/no-go plan flagged a "win rate at 21.5 %" and a structural
absence of SELL signals (0 SELL on SXRV.DE over 302 cycles). A precise audit of
`logs_prod/model_performance.db` (5306 rows, 18 outcome-labelled dates) and the
decision-engine source revealed that **both symptoms were misdiagnosed in the
plan**, and that the root causes were deeper than reported.

### 1.1 The win_rate metric measured the market, not the models

`src/adaptive_weight_manager.py` computed:

```python
win_rate = (returns > 0).mean()   # fraction of days return_1d > 0
```

`return_1d` is the **market** return, identical across models on a given date.
Over the 18 labelled dates the market fell 14 times, so this returned ~0.21 for
**every** model. The adaptive-weight subsystem was therefore steering weights on
a metric that discriminated nothing.

### 1.2 Five mechanisms produced the structural bullish bias

| # | Location | Mechanism |
|---|---|---|
| 1 | `enhanced_decision_engine.py` `MIN_CONFIDENCE_FOR_SELL` | SELL required 0.40 confidence, BUY only 0.20 — asymmetric |
| 2 | `enhanced_decision_engine.py` `SUPER_CONSENSUS_BOOST` | +0.20 boost when classic+timesfm agreed, but timesfm never emitted SELL → BUY-only inflation |
| 3 | `advanced_risk_manager.py` EXIT INERTIA | A held position needed 0.45-0.55 confidence to exit; no stop-loss bypass → CRUDP.PA drifted to -18 % with every SELL squelched to HOLD |
| 4 | `timesfm_model.py` position filter | `elif signal=="SELL" and position=="FLAT": signal="HOLD"` — the default position is FLAT, so **every** bearish vote was suppressed (0 SELL / 610 predictions) |
| 5 | four models structurally | grebenkov 610/610 BUY, sentiment/timesfm/oil_bench ~0 SELL → ~46 % of consensus weight could never vote bearish |

### 1.3 Two misdiagnoses in the plan

- **"vincent_ganne = meilleur (23.8 %)"** — false. Its `edge_buy` (mean return
  on its BUY days minus market) was **-0.0071**. The 23.8 % was its market
  up-day fraction (the artefact metric), not its performance.
- **"hmm_model = défaillant (11.7 %)"** — misleading. The 11.7 % is the same
  artefact over a smaller sample (403 obs). Its real accuracy (0.414) ≈ classic.

---

## 2. Decision

Apply eight targeted fixes (P0/P1 from the audit), all gated by tests.

### 2.1 Per-signal win_rate (P0)

`src/adaptive_weight_manager.py` — new module-level `_signal_correct_mask(df)`:
- BUY/STRONG_BUY correct iff `return_1d > 0`
- SELL/STRONG_SELL correct iff `return_1d < 0`
- HOLD/NEUTRAL correct iff `|return_1d| < 0.005`

Applied in both `calculate_model_performance` and `calculate_all_models_performance`.

### 2.2 Symmetric confidence thresholds (P0)

`src/enhanced_decision_engine.py` — `MIN_CONFIDENCE_FOR_SELL = 0.20` (was 0.40),
matching `MIN_CONFIDENCE_FOR_ACTION`. `SUPER_CONSENSUS_BOOST = 0.0` (was 0.20)
until timesfm can emit SELL symmetrically.

### 2.3 Hard-stop drawdown bypass (P0)

`src/advanced_risk_manager.py` — new `hard_stop_drawdown = 0.12` parameter.
When a held position's index drawdown from entry exceeds it, EXIT INERTIA is
bypassed and the SELL passes regardless of conviction. Configurable via
`risk_parameters.hard_stop_drawdown`.

### 2.4 TimesFM SELL reachability (P1)

`src/timesfm_model.py` — removed the `SELL → HOLD when FLAT` branch. SELL is now
a directional vote; whether to act (close a long / short) is decided downstream
by the risk manager. The `BUY → HOLD when LONG` de-churn guard is kept.

### 2.5 TensorTrade confidence cap (P1)

`src/tensortrade_model.py` — `confidence = min(probs[action], _CONFIDENCE_CAP)`
with `_CONFIDENCE_CAP = 0.75`. The PPO policy distribution collapses after
repeated in-call fine-tuning with no entropy regularization, producing ~0.88
systematically — uncalibrated. A full isotonic recalibration is deferred.

### 2.6 Reponderation by edge_buy (P1)

`src/config_weights.py` — weights rederived from `edge_buy` (return on BUY days
vs market), not the artefact metric. Notable: `llm_text` 0.21→0.12 (worst
edge_buy -0.014), `oil_bench` 0.05→0.08 (only profitable SELL, +0.009).

### 2.7 Documentation (P2)

This ADR. Plus a docstring on `sentiment_analysis.py` noting that its 0-SELL
symptom is an upstream data skew (thresholds already symmetric), not a logic bug.

---

## 3. Evidence (prod data, 29/05 → 25/06)

Measured before the fix on `logs_prod/model_performance.db`:

| Model | Poids (avant) | accuracy | precision | **edge_buy** |
|---|---|---|---|---|
| sentiment | 0.16 | 0.775* | 0.385 | +0.0166 |
| oil_bench | 0.05 | 0.697* | 0.284 | +0.0093 |
| classic | 0.13 | 0.416 | 0.269 | +0.0038 |
| timesfm | 0.20 | 0.700* | 0.216 | -0.0005 |
| llm_visual | 0.19 | 0.472 | 0.211 | -0.0016 |
| tensortrade | 0.05 | 0.443 | 0.207 | -0.0047 |
| hmm_model | 0.05 | 0.414 | 0.149 | -0.0023 |
| vincent_ganne | 0.05 | 0.231* | 0.139 | -0.0071 |
| grebenkov | 0.05 | 0.215* | 0.215 | -0.0089 |
| llm_text | 0.21 | 0.698* | 0.095 | **-0.0141** |

`*` accuracy inflated by the market being 78 % down — a model predicting
HOLD/SELL on a down market scores high without skill.

Signal distribution confirmed the bias: grebenkov 610 BUY / 0 SELL / 0 HOLD;
timesfm 88 BUY / 522 HOLD / 0 SELL; sentiment 27 BUY / 583 HOLD / 0 SELL.

---

## 4. Tests

- `tests/test_adaptive_weight_manager.py` (new) — pins per-signal win_rate: a
  model BUY on up-days scores 1.0, the inverse scores 0.0, same-market models
  are now discriminated.
- `tests/test_enhanced_decision_engine.py` — `TestBaisRemovalInvariants` pins
  symmetric thresholds and disabled boost.
- `tests/test_advanced_risk_manager.py` (new) — inertia blocks weak SELL in
  profit, hard-stop releases SELL at -18 %, threshold boundary respected.
- `tests/test_timesfm_model.py` — updated to assert SELL is emitted from FLAT.
- `tests/test_weight_alignment.py` — fixed (was broken on main, expected a
  `finacumen` weight absent from config); now asserts engine↔manager parity.
- `tests/test_tensortrade_integration.py` — updated to the cap (0.75) and
  reweight (0.04).

---

## 5. Out of scope (follow-ups)

- Isotonic/Platt recalibration of tensortrade (cap is the interim fix).
- Upstream investigation of the sentiment score never going below -0.15.
- Unifying the two divergent `regime_adjustments` dicts (engine vs manager).
- Peuplement of `return_5d` and per-regime evaluation.
- FinAcumen wiring into the real-time consensus.
- `backtest_prod.py` cache-path mismatch (reads root cache, not prod cache).

---

## 6. Reversal procedure

Each fix is independent and individually revertible:

- **win_rate**: revert the two `win_rate = float(_signal_correct_mask(...).mean())`
  lines to `(returns > 0).mean()`; remove `_signal_correct_mask`.
- **thresholds**: restore `MIN_CONFIDENCE_FOR_SELL = 0.40`,
  `SUPER_CONSENSUS_BOOST = 0.20`.
- **hard-stop**: delete the `if index_perf < -self.hard_stop_drawdown:` block.
- **timesfm**: restore the `elif signal == "SELL" ... == "FLAT": signal = "HOLD"` branch.
- **cap**: remove the `confidence = min(confidence, _CONFIDENCE_CAP)` line.
- **weights**: restore the previous `DEFAULT_BASE_WEIGHTS` dict from git history.

No database migration, no schema change, no LLM call-site change — the §2.1
dual-layer JSON defence invariant (`AGENTS.md` §2.1) is untouched.
