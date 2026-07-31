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
- **Audit PROD 2026-07-31 réalisé + 3 correctifs anti-churn/EIA/sell-guard** : verdict WARN. Investigation profonde → 3 causes racines du churn destructeur sur CRUDP.PA (3 round-trips vendus en perte le 30/07, gap BUY→SELL = 1 cycle ~30 min). **3 correctifs implémentés** : (1) EIA cache guard fraîcheur contenu (`MAX_CRUDE_IMPORTS_AGE_DAYS=70`) — le payload de 3 lignes stale passait le guard `len>=3` car seule la mtime était vérifiée ; (2) sell-loss guard lit enfin le vrai champ T212 `averagePricePaid` via helper `_get_avg_price()` (l'ancien `averagePrice` est absent du payload live → `t212_buy_cost=0` → guard neutralisé = **cause racine du churn**) ; (3) anti-churn `_evaluate_min_holding` (4h, BUY→SELL, bypassé par hard-stop). 43/43 + 67/67 tests OK, 0 régression, 8 nouveaux tests. Détails : `log.md` 2026-07-31.
- **Déploiement PROD** : `git pull` puis **supprimer `data_cache/eia/eia_crude_imports.parquet` sur la machine PROD** (forcer un re-fetch live — le cache stale actuel sinon resterait). **Pas de reset complet** ni de wipe DB nécessaire.
- **Surveiller après pull** : les logs doivent montrer (a) plus de `VENTE BLOQUÉE` sur les ventes en perte légère, (b) quelques `🛡 ANTI-CHURN: consensus SELL blocked` dans les 4h suivant un BUY, (c) un cache EIA `crude_imports` avec `latest period` récent (< 70j). Si le churn persiste (round-trips < 4h), investiguer si le `force_stop_loss` est trop souvent True (ce qui bypass l'anti-churn).
- **Review fin août** dès données suffisantes : `uv run python audit_prod_logs.py` → analyser `logs_prod/audit_report.md` → confirmer que le P&L réalisé n'est plus érodé par le churn.

## Statut des Invariants Critiques (contrôle rapide)
- [x] Défense JSON bi-couche active aux 4 sites (`<|think|>` préfixe + schema strict + suffixe). *(AGENTS.md §2.1)*
- [x] Budget 1000€/ticker (`INITIAL_BUDGETS`), pas le fallback 5000€.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
