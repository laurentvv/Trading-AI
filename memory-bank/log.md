# Journal d'Exécution (Append-Only)

> **Rôle** — Journal chronologique **en mode ajout seul**. Rien ne doit y être effacé.
> Chaque événement majeur y est empilé. Une entrée est ajoutée au début et à la fin de chaque action.
> (Historique produit → voir `changelog.md`. Tableau de bord sprint → voir `progress.md`.)
> Voir `AGENTS.md §1` pour la discipline des 4 fichiers et le protocole de gestion d'erreurs.

---

## [2026-06-30] init | Initialisation du système de mémoire déterministe (4 fichiers)
- Mise en place de la discipline des 4 fichiers dans `memory-bank/` (voir `AGENTS.md §1`).
- Création de `feature_list.json` (22 fonctionnalités cartographiées, F-01 → F-22).
- Création de `contract.md` (critères d'acceptation automatisés + protocole d'évaluation).
- Création de ce `progress.md` (tableau de bord sprint : période de validation PROD).
- Création de ce `log.md` (journal d'exécution append-only).
- Renommage de l'ancien `progress.md` (167 lignes d'historique corrections 2025-2026) → `changelog.md`.
- Refonte/allègement de `AGENTS.md` (racine) avec intégration du Principe Fondamental + directives opérationnelles.

## [2026-06-30] gen | Synchronisation de l'état initial
- `feature_list.json` : F-19 (validation PROD) marqué `in_progress` ; F-20/F-21 marqués `pending`.
- `progress.md` : objectif sprint aligné sur la période de validation 2026-05-29 → 2026-06-30.

## [2026-07-03] fix | Refonte reset_for_fresh_test.py (PR #75/#76/#77)
- Wipe par pattern (extensions gitignored) au lieu d'une liste explicite fragile.
- Bug `trading_history.db` sauté quand `cache_moved=True` corrigé (cause de désync).
- Bug `_safe_to_wipe` bloquant les sous-dossiers WIPE_DIRS de dossiers keep corrigé.
- Fallback copy+truncate pour fichiers verrouillés Windows (scheduler.log).
- Ledger Gemini wipé par défaut (démo) ; flag `--keep-quota-ledger` pour PROD payante.

## [2026-07-09] fix | 4 bugs PROD : phantom trades, T212 precision, win_rate, EIA (PR #78/#79)
- **#1 CRITIQUE** : `write_db=not is_t212` — la simulation n'écrit plus en DB en mode T212.
- **#2** : table `TICKER_QUANTITY_PRECISION` (OD7Fd_EQ=2, SXRVd_EQ=4, fallback=2).
- **#3** : sentinelle `-1.0` cohérente pour win_rate "non calculable" ; 72 fausses alertes éliminées.
- **#4** : `audit_prod_logs.py` utilise la colonne `period` pour EIA (au lieu de `df.index` → 1970).
- Nouveau script `clean_phantom_trades.py` pour nettoyage ciblé post-fix.
- Validation PROD : 0 transaction fantôme, 0 fausse alerte, state synced broker, 0 nouvelle erreur.

## [2026-07-09] gen | Mise à jour doc (AGENTS.md §2.2/§6/§7, changelog.md, log.md)
- `AGENTS.md §2.2` : 3 nouveaux invariants (write_db, TICKER_QUANTITY_PRECISION, win_rate sentinel).
- `AGENTS.md §6.1/§6.2` : bugs July 2026 résolus documentés ; ADR-002 Suite renuméroté.
- `AGENTS.md §7` : ajout `reset_for_fresh_test.py` et `clean_phantom_trades.py`.
- `changelog.md §5` : 2 entrées corrections récentes (reset refonte + 4 bugs).
- Phase de validation longue durée lancée (jusque fin juillet) avant passage PROD.

## [2026-07-15] fix | Audit PROD 2026-07-15 : 3 bugs (risk manager, biais SELL, EIA)
- **Audit initial** : `uv run python audit_prod_logs.py` → verdict WARN. Cohérence DB/broker parfaite (bug phantom-trades confirmé résolu), fix win_rate/precision effectifs. Mais 3 anomalies de comportement :
  - `Risk_Level = VERY_HIGH` sur 100% des 294 cycles → neutralise SXRV.DE (147/147 Risk_Adjusted=HOLD).
  - 0 signal SELL sur 294 cycles malgré ~400 votes SELL individuels.
  - `eia_crude_imports.parquet` dégénéré (1 ligne @ 2026-04-01).
- **Causes racines confirmées par reproduction sur logs_prod/data_cache** :
  - #1 : `advanced_risk_manager.py` applique des seuils échelle volatilité (0.01–0.04) à un score composite 0–1 → tout score > 0.04 = VERY_HIGH. Liquidity_risk structurel ~0.74 (pattern_risk faux pour ETF).
  - #2 : abstention HOLD non renormalisée + vincent_ganne 100% BUY sur non-oil + seuil SELL -0.15 inatteignable (score bearish max -0.139).
  - #3 : `get_crude_imports` sans facets → payload 1 ligne ; TTL basé sur mtime masque le problème.
- **Action en cours** : 4 correctifs ciblés + test de régression PROD (plan approuvé, approche « ciblée et sûre »).

## [2026-07-15] fix | 4 correctifs implémentés et validés (96/96 tests OK)
- **#1 Risk manager** (`advanced_risk_manager.py:94-100`) : seuils rescalés échelle composite 0-1 (VERY_LOW 0.20 → VERY_HIGH inf), au lieu de l'échelle volatilité (0.01-0.04) qui forçait VERY_HIGH sur 100% des cycles. SXRV.DE (score ~0.42) sort de VERY_HIGH → ses BUY ne sont plus neutralisés.
- **#2 Liquidity risk** (`advanced_risk_manager.py:236-242`) : `pattern_risk = 1-|corr(volume,returns)|` plafonné à 0.5 et sous-pondéré (40% vs 60% pour volume_risk). Corrige l'inflation structurelle des ETF (~0.74 → ~0.50).
- **#3a Consensus renormalisé** (`enhanced_decision_engine.py:511-540`) : le score pondéré est normalisé sur le poids des modèles votants (non-HOLD), plus sur tous les modèles. Les abstentions ne diluent plus le score. **Quorum guard** : si < 25% du poids a voté, fallback sur score brut (évite qu'un modèle isolé déclenche un signal fort — 47/294 cycles PROD concernés).
- **#3b Seuil SELL** (`enhanced_decision_engine.py:344`) : -0.15 → -0.10 (rend SELL atteignable ; le cycle le plus bearish PROD passait de -0.139 à -0.32 avec la renormalisation).
- **#3c Vincent Ganne** (`enhanced_trading_example.py:651-662`) : désactivé sur les tickers non-oil (vote BUY/STRONG_BUY 147/147 sur SXRV.DE = bruit macro sans lien avec l'ETF equity). `effective_vg_indicators = None`.
- **#4 EIA crude_imports** (`eia_client.py:134-170`) : ajout `facets[process][]=MCO` + tri asc ; refus de cacher une payload < 3 lignes (empêche le mtime de masquer une dégénérescence).
- **Test de régression** `tests/test_prod_regression.py` (6 tests) : rejoue les vraies données logs_prod/ → assert SXRV.DE < 50% VERY_HIGH, SELL atteignable, EIA pas de cache dégénéré.
- **Validation** : 96/96 tests OK (suite mockée complète + régression). `test_bias_removal.py` corrigé par le quorum guard (1 SELL seul → HOLD, pas STRONG_SELL).
- **Rappel PROD** : supprimer `logs_prod/data_cache/eia/eia_crude_imports.parquet` après `git pull` pour forcer le re-fetch EIA.
