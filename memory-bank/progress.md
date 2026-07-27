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
- **Audit PROD 2026-07-27 réalisé + correctif win_rate** : le gate dur `win_rate < 45%` (`adaptive_weight_manager.py`) neutralisait 8 modèles sur 10 (win_rate structurellement < 45% sur ETF peu volatils car la métrique ADR-002 inclut une dead-zone). Remplacé par une **rampe douce** `WIN_RATE_SOFT_FLOOR=0.25` → `CEIL=0.50` (facteur linéaire) — préserve la diversité de l'ensemble au lieu de l'effondrer à 3 votants. 36/36 tests OK, 0 régression. Détails : `log.md` 2026-07-27, AGENTS.md §6.3.
- **Reset complet PROD décidé** : vu l'accumulation de bugs (gate non-commité, EIA dégénéré, modèles 404 — ces 2 derniers déjà corrigés par le `git pull` 2026-07-27 50906b6), exécuter après merge du fix win_rate : `git pull` PROD puis `uv run python reset_for_fresh_test.py --dry-run` et `--yes` (DEMO ; `--keep-quota-ledger` si PAID). Premier cycle plus lent (re-download/re-train).
- **Surveiller après reset** : logs `Réduction de ...` (INFO) au lieu de `🚨 Bloquage` ; ~7 votants significatifs ; diversité restaurée ; SELL doit rester atteignable ; `Risk_Level` doit varier (pas 100% VERY_HIGH).
- **Review fin août** dès données suffisantes : `uv run python audit_prod_logs.py` → analyser `logs_prod/audit_report.md` (Sharpe / MaxDD / Win Rate / Alpha par ticker) → décider ajustements de poids.

## Statut des Invariants Critiques (contrôle rapide)
- [x] Défense JSON bi-couche active aux 4 sites (`<|think|>` préfixe + schema strict + suffixe). *(AGENTS.md §2.1)*
- [x] Budget 1000€/ticker (`INITIAL_BUDGETS`), pas le fallback 5000€.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
