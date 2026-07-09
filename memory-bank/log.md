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
