# Audit Production logs_prod/ — 2026-06-23

**Verdict global:** ⚠️ AVERTISSEMENTS

_Genere par `audit_prod_logs.py`. Valide l'integrite de tous les fichiers de production, la coherence des bases SQLite, la fraicheur des caches, execute un backtest corrige (source prod) et verifie la chaine d'outils FinAcumen._

---
### ℹ️ Catalogue des fichiers

- **Fichiers:** 114
- **Taille totale:** 47.11 Mo
- **Periode courante (date du jour):** 2026-06-23

| Extension | Nombre |
|-----------|--------|
| `.json` | 40 |
| `.pkl` | 39 |
| `.parquet` | 20 |
| `.db` | 4 |
| `.png` | 3 |
| `.txt` | 3 |
| `.log` | 2 |
| `.md` | 1 |
| `.csv` | 1 |
| `.zip` | 1 |

### ✅ Journal de trading (CSV)

- **Lignes:** 572
- **Periode:** 2026-06-01 -> 2026-06-23
- **Colonnes attendues presentes:** oui

| Ticker | Total | BUY | SELL | HOLD | Conf moy |
|--------|-------|-----|------|------|----------|
| SXRV.DE | 284 | 283 | 0 | 1 | 36.1% |
| CRUDP.PA | 288 | 142 | 3 | 143 | 21.3% |

### ✅ Bases SQLite

| Base | Tables (lignes) |
|------|-----------------|
| trading_history.db | transactions=7, sqlite_sequence=2, portfolio_history=35, model_signals=0 |
| performance_monitor.db | realtime_metrics=572, sqlite_sequence=2, performance_alerts=14, daily_performance=0 |
| model_performance.db | model_performance_history=4964, sqlite_sequence=1, model_performance_summary=0 |
| finacumen_memory.db | memories=3, sqlite_sequence=1 |

**Transactions T212 reelles:** 7
| Date | Ticker | Type | Qte | Prix | Source |
|------|--------|------|-----|------|--------|
| 2026-05-29 00:00:00 | SXRV.DE | BUY | 0.6644 | 1503.60 | STRONG_BUY |
| 2026-06-01 00:00:00 | CRUDP.PA | BUY | 76.7694 | 13.01 | STRONG_BUY |
| 2026-06-01 09:59:49 | SXRV.DE | BUY | 0.6332 | 1500.40 | IA_HYBRID_T212 |
| 2026-06-01 10:02:57 | CRUDP.PA | BUY | 72.8200 | 13.05 | IA_HYBRID_T212 |
| 2026-06-08 09:35:22 | CRUDP.PA | SELL | 72.8200 | 13.66 | IA_HYBRID_T212 |
| 2026-06-09 00:00:00 | CRUDP.PA | BUY | 74.4522 | 13.42 | BUY |
| 2026-06-09 08:36:02 | CRUDP.PA | BUY | 70.8000 | 13.42 | IA_HYBRID_T212 |


### ⚠️ Caches parquet (prix + EIA)

| Fichier | Lignes | Periode / derniere date | Couverture juin 2026 |
|---------|--------|--------------------------|----------------------|
| SXRV_DE_max_with_vix.parquet (SXRV.DE) | 1268 | 2021-06-23 -> 2026-06-23 | 17 bars |
| CRUDP_PA_max_with_vix.parquet (CRUDP.PA) | 1279 | 2021-06-23 -> 2026-06-23 | 17 bars |
| CL=F_max_with_vix.parquet | 1257 | 2021-06-23 -> 2026-06-23 | 16 bars |
| ^NDX_max_with_vix.parquet | 1254 | 2021-06-23 -> 2026-06-22 | 15 bars |
| EIA (10 fichiers) | - | voir detail | - |


**EIA potentiellement stale:** eia_brent_spot.parquet@1970-01-01, eia_crude_imports.parquet@1970-01-01, eia_inventories.parquet@1970-01-01, eia_refinery_util.parquet@1970-01-01, eia_steo_BREPUUS.parquet@1970-01-01

### ✅ Caches JSON (search_queries)

- **search_queries:** 35 fichiers
- **Plus recent cached_at:** 2026-06-23T08:36:06.127150

### ✅ Modeles (pickle + TensorTrade)

- **Modeles .pkl:** 39
- **TensorTrade:** last_trained=2026-06-23T10:04:57.542956, timesteps=207500, obs_shape=[20]

### ✅ Backtest corrige (source prod)

- **Periode:** 2026-06-01 -> 2026-06-23
- **Signaux journaliers agreges:** 36
- **Source prix:** `logs_prod/data_cache/` (cache prod, a jour)

| Ticker | Strategie | Buy&Hold | Alpha | Sharpe (strat) | Win |
|--------|-----------|----------|-------|----------------|-----|
| SXRV.DE | -0.88% | -0.88% | +0.00% | -0.44 | n/a |
| CRUDP.PA | -15.39% | -15.39% | +0.00% | -5.24 | n/a |

> **Note:** `backtest_prod.py` original lit `data_cache/` (repo root, fin 2026-05-27) et produit des tables vides car aucune date de juin n'est couverte. Cet audit lit `logs_prod/data_cache/` (cache prod, fin 2026-06-23), d'ou des resultats non vides.

### ✅ FinAcumen — fichiers d'etat

| Ticker | Date | Status | Signal | Conf | Analyse |
|--------|------|--------|--------|------|---------|
| CRUDP.PA | 2026-06-23 | **success** | HOLD | 0.75 | The stock CRUDP.PA is currently trading at 11.5, which is below its 50 |
| SXRV.DE | 2026-06-23 | **success** | BUY | 0.85 | The stock SXRV.DE is showing strong bullish signals across multiple in |

_Trajectory trajectory_CRUDP.PA.txt: 1 observations_

_Trajectory trajectory_SXRV.DE.txt: 1 observations_

### ✅ FinAcumen — preuve des outils (post-correction)

- `CRUDP.PA` latest -> close=11.5, rsi=38.260959846751526, sma_50=12.88006004333496, sma_200=8.772809908390045 -> OK
- `SXRV.DE` latest -> close=1488.0, rsi=55.35422525491687, sma_50=1419.4800048828124, sma_200=1271.4430041503906 -> OK
- Sandbox (code type LLM, sans import): sortie=`HOLD` -> OK
- Sandbox bloque toujours `import os`: OK

> **Convergence LLM complete verifiee:** la chaine `finacumen_main.py` a etre executee en live (Ollama + gemma-4-12b) et a produit un `status: success` (HOLD 0.75 sur CRUDP.PA, BUY 0.85 sur SXRV.DE). Re-execution: `uv run python src/finacumen_main.py --ticker CRUDP.PA`
