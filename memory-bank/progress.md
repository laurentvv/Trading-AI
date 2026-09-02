# État d'Avancement du Sprint

> **Rôle** — Tableau de bord **macroscopique du sprint en cours**. Permet à l'agent de savoir instantanément
> ce qu'il fait s'il doit redémarrer. Mis à jour à la fin de chaque itération de la boucle.
> (Historique produit → voir `changelog.md`. Journal d'exécution agent → voir `log.md`.)
> Voir `AGENTS.md §1` pour la discipline des 4 fichiers.

## Objectif Actuel
- [ ] **Sprint « GO-gates PROD » (2026-08-19 →)** : remédier les 7 bloquants de l'audit indépendant
  (`docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md`) pour permettre un **nouveau run de 30 jours en démo T212**
  mesuré correctement (equity réelle, P&L fills réels, stops broker). Le passage en compte réel reste
  conditionné aux résultats de ce run (voir `docs/COMPARAISON_AUDIT_INDEPENDANT_vs_READINESS_2026-08-19.md`).

## Jalons de l'Itération

### Sprint GO-gates (en cours)
- [x] Audit indépendant complet + confrontation avec le rapport préexistant (NO-GO, 7 bloquants).
- [x] Plan de remédiation approuvé (seuils préservés ×1/0.8 ; TP fixe +8 % + SL mouvant ratchet peak×0.90 ; scheduler self-contained).
- [x] Contrat gelé (`contract.md`, 29 critères) + features F-30→F-36 déclarées.
- [x] Gate 1 (F-30) : timeout + idempotence ordres — `post_order_market` avec réconciliation broker.
- [x] Gate 2 (F-31) : SL mouvant + TP broker + ratchet cancel/replace + sync stops.
- [x] Gate 3 (F-32) : fill confirmé + prix réel (`averagePricePaid`).
- [x] Gate 4 (F-33) : vol quotidienne + seuils préservés.
- [x] Gate 5 (F-34) : macro synthétique interdite, TTL 7 j, staleness prix 3 j.
- [x] Gate 6 (F-35) : verrou d'instance, boucle résiliente, rattrapage brief, .bat superviseur.
- [x] Gate 7 (F-36) : equity FIFO + colonne T212_Equity + monitoring réel.
- [x] Suite pytest complète : **252/252 PASS** (+56 nouveaux tests, 0 échec).
- [x] Docs : `docs/PLAN_RUN_DEMO_30J.md`, `TRADING212_API_GUIDE.md`, `AGENTS.md` §2.2/§3, `CHANGELOG.md`.
- [ ] Sonde démo `tests/check_t212_stops.py` (feu vert utilisateur requis).
- [x] Smoke cycle démo + reset propre + lancement run 30 jours (`docs/PLAN_RUN_DEMO_30J.md`) — démarré 2026-08-19 16:22.

### Migration TimesFM 3.0 + reset run 2 (2026-09-02, branche `TimesFM` — DEV validé, PROD en attente)
- [x] Wrapper `src/timesfm_model.py` : API `timesfm3.TimesFM3Forecaster` (`google/timesfm-3.0-pytorch`), médiane = signal (seuils inchangés), 9 quantiles en métadonnées, contexte 2048, retry d'init dans `predict()`.
- [x] Fin du vendoring : `setup_timesfm.py` + `.gitmodules` + `[tool.uv.sources]` supprimés ; `pyproject.toml` → `timesfm>=3.0.1,<4` (PyPI) ; `check_setup()` → `find_spec("timesfm3")`.
- [x] Fix bonus : `scripts/backtest_ensemble_10y.py` (appel `predict()` invalide → TimesFM jamais compté).
- [x] `reset_for_fresh_test.py` étendu : détection artefacts runtime `logs_prod/` + `--include-logs-prod`/`--keep-logs-prod` (trou de l'incident 2026-08-24) ; `tests/test_reset_fresh_test.py` (11 tests).
- [x] Smoke script `tests/smoke_timesfm3.py` (pré-chauffage ~1,3 Go par machine, garde-fou timing 2048).
- [x] Validation DEV : **282 passed / 3 skipped** ; smoke réel PASS (download 118 s, non gated, CPU 0.38 s @2048) ; cycle `--simul` avec TimesFM 3.0 opérationnel (init cpu + prédiction).
- [x] Docs : `docs/PLAN_MIGRATION_TIMESFM3_PROD.md` (runbook), README/AGENTS/QWEN/GEMINI/SYSTEM_SUMMARY/i18n (2.5→3.0), AGENTS.md §2.2 invariant TimesFM 3.0.
- [ ] **Exécuter le runbook sur PROD** (`docs/PLAN_MIGRATION_TIMESFM3_PROD.md` §1→§10) : reset compte démo T212 → git pull + `uv sync` + suppression `vendor/` → HF_TOKEN → smoke PROD → `reset_for_fresh_test.py --yes --include-logs-prod` → relance scheduler → **run 2 de 30 jours** (GO/NO-GO = lancement + 30 j).
- [x] Merge `TimesFM` → `main` + push (accord utilisateur 2026-09-02), prêt pour `git pull` sur PROD.

### Remédiation audit J+13 (2026-09-01, validé live — commité ecb5368)
- [x] F1 CRITIQUE realized P&L : `_fifo_pnl` trie désormais les fills par date (l'API renvoie du + récent au + ancien ; le SELL 20/08 était matché contre le BUY postérieur du 28/08 → realized -1.50 € au lieu de +1.58 €). Validé live : +1.5760 € = chiffre broker. Le state se self-corrige au 1er cycle.
- [x] A5 429 : positions/order-history routés via `safe_request` (retry 429 backoff) — fini la sync annulée au premier 429.
- [x] A3 hygiène données : labels Target* masqués sur lignes gelées du feed (CRUDP.PA trading-ticker 82 % gelé 2022-2025) ; lignes conservées pour les fenêtres MA_200 ; no-op sur ^NDX/CL=F/SXRV.
- [x] A2 classic SELL permanent sur CRUDP.PA : diagnostic complet — opinion du modèle WTI (données saines), pas un bug ; défaut structurel = pas de colonne ticker dans l'évaluation des perfs → recommandation post-run (schéma + évaluation par (ticker, modèle)).
- [x] A4 vincent_ganne N/A : désactivation volontaire confirmée (audit juillet 2026).
- [x] Tests : **271/271 PASS** ; AGENTS.md §2.2 enrichi (FIFO chronologique, labels gelés).
- [ ] **Redémarrer le scheduler** (`start_scheduler.bat` depuis `logs_prod/`) — arrêté ~14:42 le 01/09 pour l'audit.
- [ ] Committer les correctifs J+13.

### Remédiation audit J+5 (2026-08-24, en cours de validation par le run)
- [x] C1 vente débloquée : stop annulé AVANT la vente (actions réservées), fallback `quantity`, re-protection d'urgence si échec (`t212_executor._execute_sell_order`).
- [x] C2 fill SELL : filtre `side=="SELL"` dans `_confirm_fill` + réconciliation prix/cash (`_reconcile_sell_fill_price`, le cash est la vérité terrain).
- [x] C3 fetchs avales : `get_t212_positions`/`get_t212_order_history` → `None` sur échec ; sync annulée, exécution T212 avortée si état broker inconnu.
- [x] C4 poids adaptatifs : `WIN_RATE_MIN_SAMPLES=20` (`n_observations` sur `ModelPerformance`) — plus de zerout sur micro-échantillons.
- [x] C5 council : SQL sur les vraies tables (`model_performance_history`, `daily_performance`, `performance_alerts`) + correction directionnelle des issues.
- [x] C6 masquage des secrets dans les stderr loggés (`_redact_secrets`).
- [x] C7 circuit breaker EIA crude_imports (12 h sur contenu périmé).
- [x] `logs_prod/model_performance.db` pollué (ère pré-reset, win_rates 0-1 %) → sauvegardé en `.bak-2026-08-24` et réinitialisé.
- [x] Tests : **269/269 PASS** (+17 dans `tests/test_prod_fixes_2026_08_24.py`, rampe win-rate portée à 25 obs).
- [ ] Validation en conditions réelles : premier cycle post-fix (attendu 2026-08-25 08:30) — guetter `🔓 Annulation préalable`, poids restaurés, plus de "Réduction forte" en rafale.

### Historique (sprints précédents — acquis)
- [x] Migration NexusAI-Client complète (2026-08-18), cycle ~93 s, 193→196 tests verts.
- [x] Audit GO conditionnel précédent (rapport `AUDIT_PROD_READINESS_TRADING212.md`) — **contredit par
  l'audit indépendant (NO-GO)** ; voir la confrontation pour la réconciliation des deux lectures.

### Suivis (post-sprint, hors périmètre GO-gates)
- [ ] M1 : échec classic → vote SELL fantôme (neutraliser en HOLD).
- [ ] M2 : distinction HOLD technique vs HOLD modèle dans la base perf.
- [ ] M3 : cycle de vie PPO TensorTrade (par ticker, réentraînement planifié).
- [ ] M4 : dé-doublonner le double comptage du council.
- [ ] M7 : UNIQUE (date, modèle) + seuil en jours dans les poids adaptatifs ; drawdown série inversée.
- [ ] Optimisation des poids par grid search (`backtest_prod.py`).

## Prochaine Action Immédiate
- **Exécuter le runbook migration PROD** (`docs/PLAN_MIGRATION_TIMESFM3_PROD.md`) : le run 1 est
  clos (3 ordres/13 j, critère ≥20 round-trips inatteignable, scheduler arrêté depuis le 01/09).
  Après reset (compte démo T212 + local `--include-logs-prod`), **run 2 de 30 jours avec TimesFM 3.0**.
- Supervision post-lancement selon `docs/PLAN_RUN_DEMO_30J.md` §4 ; GO/NO-GO = lancement + 30 j.
- À guetter au 1er cycle : `TimesFM 3.0 prediction: ...` dans trading.log, journal régénéré,
  T212_Equity repartant à 1000 €/ticker (pas d'héritage FIFO de la run 1).

## Statut des Invariants Critiques (contrôle rapide)
- [x] Architecture NexusAI Cloud active (auto_fallback & auto_fallback_vision) avec validation JSON stricte.
- [x] Budget 1000 €/ticker (`INITIAL_BUDGETS`), pas le fallback 5000 €.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
- [x] `write_db = not is_t212` — seul l'exécuteur broker écrit en DB (à préserver pendant tout le sprint).
