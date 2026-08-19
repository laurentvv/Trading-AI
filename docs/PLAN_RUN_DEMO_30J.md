# Plan de run — 30 jours de démo T212 (validation GO-gates)

**Date :** 2026-08-19 — **Statut : prêt à lancer** (après sonde broker, voir §2)
**Contexte :** remédiation complète des 7 GO-gates de l'audit indépendant
(`AUDIT_PROD_INDEPENDANT_2026-08-19.md`). Ce run de 30 jours en **démo** est la
phase de preuve demandée avant toute décision de passage en compte payant.

---

## 1. Ce qui a changé (et pourquoi ce run est comparable et mesurable)

| GO-gate | Correctif en place | Effet mesurable attendu |
|---|---|---|
| C1 — timeout/idempotence | `post_order_market` : timeout 15 s, réconciliation position broker avant tout retry | 0 double-ordre possible ; traces « AUCUN re-POST » dans `trading.log` en cas de réponse perdue |
| C2 — protections broker | TP +8 % attaché à l'ordre ; stop GTC dédié à −10 % du fill réel ; **ratchet** cancel/replace vers peak×0.90 à chaque cycle | Chaque position ouverte a un stop visible dans l'app T212 ; le stop ne descend jamais |
| C3 — fill réel | État/DB écrits après observation du fill ; prix `averagePricePaid` | P&L DB ≈ P&L broker (l'écart de ~9,9 € du run précédent doit disparaître) |
| C4 — vol quotidienne | Vol 20 jours non annualisée ; seuils recalés ×1/0.8 | L'amortissement ×0.8 ne s'applique qu'en VRAIE haute volatilité |
| C5/C6 — données | Macro synthétique supprimée ; TTL macro 7 j ; fallback prix refusé au-delà de 3 j | Logs « REFUSING to fabricate » / « REFUSING the fallback » si sources mortes (et cycle sauté, pas de trade aveugle) |
| I1/I2 — scheduler | Verrou `scheduler.lock` + boucle résiliente + relance .bat + rattrapage brief | 1 instance max ; crash → relance en 30 s ; brief rattrapé dans la journée |
| G7 — mesure | Colonne `T212_Equity` (budget + réalisé FIFO + latent) ; monitoring réel | Courbe d'equity exploitable dans `trading_journal.csv` et `performance_monitor.db` |

Seuils de décision **volontairement préservés** (comportement effectif identique :
BUY 0.15 / SELL −0.125 / STRONG ±0.4375/−0.5625) pour que ce run mesure les
correctifs d'exécution sans confondre avec un changement de stratégie.

## 2. Pré-requis avant lancement (dans l'ordre)

1. **Sonde broker démo** (avec feu vert explicite, compte démo uniquement, ~0,15 € d'empreinte) :
   ```powershell
   uv run python tests/check_t212_stops.py
   ```
   Valide sur le compte réel de démo : ordre stop dédié, visibilité dans
   `/equity/orders`, DELETE (ratchet), attachement `takeProfit`.
   → Consigner le bilan dans `TRADING212_API_GUIDE.md`. Si le `takeProfit`
   attaché est refusé (400) : comportement normal, le fallback « ordre nu +
   stop dédié » est déjà codé et testé.
2. **Vérifier `.env.t212`** : `T212_ENV="demo"` (le scheduler affiche DEMO au
   démarrage et chaque exécution loggue `EXÉCUTION IA TRADING 212 (DEMO)`).
3. **Suite de tests** (doit être 252/252) :
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp
   ```
   (harness live ignorés : `test_crawl4ai`, `check_*`, `bench_*`, `run_short_backtest`)

## 3. Reset propre et lancement

```powershell
# 1. Vérifier ce qui sera effacé (réversible : tout va dans reset_backup/<timestamp>)
uv run python reset_for_fresh_test.py --dry-run

# 2. Appliquer (efface DBs, journal CSV, caches, state T212 ; PPO re-apprendra au 1er cycle — connu et accepté)
uv run python reset_for_fresh_test.py --yes

# 3. Lancer le scheduler supervisé (relance auto après crash, verrou anti-doublon)
.\start_scheduler.bat
```

**Important :**
- Lancer **une seule** instance (le verrou refuse la 2ᵉ : message « Instance
  dupliquée refusée »).
- Arrêt propre : **Ctrl+C** dans la fenêtre (code 0 → le superviseur s'arrête).
- Machine allumée en continu recommandé ; si coupure : relancer le `.bat`,
  le rattrapage du brief et le verrou périmé (2 h) gèrent la reprise.

## 4. Checklist de supervision

**Quotidienne (2 min)** :
- [ ] `scheduler.log` : cycles 08:30→18:00 présents, aucune « Instance dupliquée ».
- [ ] `trading.log` : pas de `CRITICAL` non expliqué ; les « REFUSING » données
      sont normaux uniquement si les sources sont réellement mortes.
- [ ] `trading_journal.csv` : colonne `T212_Equity` cohérente (pas de 1000 figé,
      pas de saut position/cash).
- [ ] Positions ouvertes : un stop-loss visible dessous dans l'app T212, qui
      remonte quand la position gagne (ratchet).

**Hebdomadaire (15 min)** :
- [ ] Rapport council du samedi généré (`docs/council_reports/`).
- [ ] Rapprochement P&L : `SELECT` sur `trading_history.db` vs historique T212 —
      l'écart doit être ~0 (prix de fill réels désormais).
- [ ] `performance_monitor.db` : `portfolio_value` suit l'equity (plus de
      1000 constant), `active_positions`/`cash_balance` réels.

**Kill-switch (si anomalie)** :
1. Ctrl+C sur le scheduler (ou `Stop-Process -Name python -Force`).
2. Vendre manuellement les positions dans l'app T212 (les stops broker restent
   actifs en attendant — c'est le but).
3. Révoquer la clé API dans les paramètres T212 si nécessaire.

## 5. Critères de succès du run (décision GO/NO-GO PROD à J+30)

| Critère | Seuil |
|---|---|
| Intégrité exécution | 0 double-ordre, 0 position sans stop broker, écart DB/broker < 0,5 % |
| Disponibilité | ≥ 95 % des cycles planifiés exécutés (verrou + relance effectifs) |
| Données | 0 décision sur cache > 3 j ou macro synthétique (interdit par code) |
| Mesure | Courbe d'equity continue sur 30 jours pour les 2 tickers |
| Performance | **P&L net > 0 après spread** sur ≥ 20 round-trips (sinon : pas d'edge démontré → pas de PROD) |
| Qualité signaux | Win rate et drawdown cohérents avec les seuils ; part de HOLD documentée |

Un seul critère manquant = le passage en compte payant reste reporté ; les
critères d'intégrité (4 premiers) sont rédhibitoires quelle que soit la
performance.

## 6. Limites connues et acceptées pour ce run (hors périmètre GO-gates)

- L'échec du modèle classic vote toujours SELL 0.5 (M1) — non corrigé
  volontairement pour ne pas changer le comportement mesuré.
- Les HOLD techniques polluent toujours le win rate des poids adaptatifs (M2/M7).
- Le PPO TensorTrade se réentraîne au premier cycle (reset) et reste partagé
  entre tickers (M3).
- Le council reste compté deux fois (M4).

Ces limites sont celles listées « suivis post-sprint » dans
`memory-bank/progress.md` ; elles n'affectent pas la sécurité capitalistique
corrigée par les 7 GO-gates.
