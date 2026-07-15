# État d'Avancement du Sprint

> **Rôle** — Tableau de bord **macroscopique du sprint en cours**. Permet à l'agent de savoir instantanément
> ce qu'il fait s'il doit redémarrer. Mis à jour à la fin de chaque itération de la boucle.
> (Historique produit → voir `changelog.md`. Journal d'exécution agent → voir `log.md`.)
> Voir `AGENTS.md §1` pour la discipline des 4 fichiers.

## Objectif Actuel
- [ ] **Période de validation PROD** (2026-05-29 → 2026-06-30) : confirmer que tous les modèles performent correctement sur de vraies transactions T212, puis ajuster les poids si nécessaire. *(F-19)*

## Jalons de l'Itération

### Période de validation (en cours)
- [x] Fresh start PROD — toutes les DBs wipeées (`trading_history.db`, `model_performance.db`).
- [x] TensorTrade PPO persistence déployée (`data_cache/tensortrade/ppo_model.zip`, env 10 features).
- [x] Premier cycle PROD validé (2026-05-29) : SXRV.DE + CRUDP.PA en ~488 sec, 0 référence Kronos.
- [x] FinAcumen réparé (2026-06-23) : convergence `status: success` (était `timeout` à chaque run).
- [x] Weekend Council déployé + code review critique (2026-06-28) : 11ème voix (9.5%) active.
- [x] FinAcumen : 6 bugs corrigés (`src/core/tools.py`, `src/agents/solver.py`).
- [ ] **Review fin juin 2026** : évaluer Sharpe, win rate, précision par modèle.
- [ ] Décider d'éventuels ajustements de poids (AdaptiveWeightManager).

### Suivis (post-validation)
- [ ] Optimisation des poids par grid search (`backtest_prod.py`). *(F-20)*
- [ ] Recalibration isotonic TensorTrade (cap intérimaire selon ADR-002). *(F-21)*
- [ ] Corriger `backtest_prod.py` : lire `logs_prod/data_cache/` au lieu du cache racine périmé.
- [ ] Source de prix live alternative pour ETFs sans position T212 ouverte (SXRV.DE).
- [ ] Synchroniser les traductions i18n (9 langues) avec les mises à jour README.

## Prochaine Action Immédiate
- **Audit PROD 2026-07-15 réalisé** : 3 bugs de comportement trouvés (risk manager VERY_HIGH permanent, 0 SELL, EIA stale) → **4 correctifs implémentés et validés** (96/96 tests OK, voir `log.md` 2026-07-15). Après `git pull` PROD : supprimer `logs_prod/data_cache/eia/eia_crude_imports.parquet` puis relancer. Surveiller les prochains cycles : SXRV.DE doit pouvoir trader, des SELL doivent apparaître, le `Risk_Level` doit varier (pas 100% VERY_HIGH).
- **Review fin juin/août** dès données suffisantes : `uv run python audit_prod_logs.py` → analyser `logs_prod/audit_report.md` (Sharpe / MaxDD / Win Rate / Alpha par ticker) → décider ajustements de poids.

## Statut des Invariants Critiques (contrôle rapide)
- [x] Défense JSON bi-couche active aux 4 sites (`<|think|>` préfixe + schema strict + suffixe). *(AGENTS.md §2.1)*
- [x] Budget 1000€/ticker (`INITIAL_BUDGETS`), pas le fallback 5000€.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
