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
- [x] **Migration NexusAI-Client (2026-08-18)** : Remplacement complet d'Ollama et de toutes les IA locales par `nexusai-client`. Vitesse par cycle : **~93s** (au lieu de 10-15 min). 193/193 tests unitaires OK.
- [x] **Audit Global pour passage en PROD Réelle (2026-08-19)** : Audit complet (code, flux, modèles, risques, broker T212, base de données). Verdict : **GO CONDITIONNEL (APPROUVÉ)**. Rapport généré dans `AUDIT_PROD_READINESS_TRADING212.md`.
- [ ] **Déploiement Compte Réel Trading 212** : Configuration de la clé API Live dans `.env.t212` et lancement du scheduler.

### Suivis (post-validation)
- [ ] Optimisation des poids par grid search (`backtest_prod.py`). *(F-20)*
- [ ] Recalibration isotonic TensorTrade (cap intérimaire selon ADR-002). *(F-21)*
- [ ] Corriger `backtest_prod.py` : lire `logs_prod/data_cache/` au lieu du cache racine périmé.
- [ ] Source de prix live alternative pour ETFs sans position T212 ouverte (SXRV.DE).
- [ ] Synchroniser les traductions i18n (9 langues) avec les mises à jour README.

## Prochaine Action Immédiate
- **Lancement en Production Réelle (Compte Payant T212)** :
  1. Suivre le protocole de déploiement (Étape 7 du rapport d'audit).
  2. Configurer la clé API Live dans `.env.t212`.
  3. Lancer le scheduler automatique : `uv run schedule.py`.

## Statut des Invariants Critiques (contrôle rapide)
- [x] Architecture NexusAI Cloud active (auto_fallback & auto_fallback_vision) avec validation JSON stricte (`_find_dict_with_keys`).
- [x] Budget 1000€/ticker (`INITIAL_BUDGETS`), pas le fallback 5000€.
- [x] Cache staleness 1 jour, cycle timeout 40 min, orphan-thread lock par ticker.
