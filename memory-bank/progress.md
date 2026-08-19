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
- [ ] Smoke cycle démo + reset propre + lancement run 30 jours (`docs/PLAN_RUN_DEMO_30J.md`).

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
- **Rien — le run de 30 jours tourne** (départ 2026-08-19 16:22, décision GO/NO-GO ≈ 2026-09-18).
- Supervision quotidienne/hebdo selon `docs/PLAN_RUN_DEMO_30J.md` §4 (checklist).
- À guetter dans les logs : `🔐 Ratchet: stop broker ... remonté` (premier plus-haut),
  l'equity qui dévie de 1000 € selon le P&L, et l'absence de CRITICAL non expliqué.
- Post-run : évaluer les critères §5 de PLAN_RUN_DEMO_30J, puis seulement discuter
  du passage en compte réel T212.

## Statut des Invariants Critiques (contrôle rapide)
- [x] Architecture NexusAI Cloud active (auto_fallback & auto_fallback_vision) avec validation JSON stricte.
- [x] Budget 1000 €/ticker (`INITIAL_BUDGETS`), pas le fallback 5000 €.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
- [x] `write_db = not is_t212` — seul l'exécuteur broker écrit en DB (à préserver pendant tout le sprint).
