# Contrat de Validation — Sprint « GO-gates PROD » (2026-08-19)

> **Rôle** — Contrat de validation technique négocié entre la planification et l'évaluation.
> Liste d'assertions strictes et testables. **Figé juste avant l'écriture de la première ligne de code** :
> une fois le code démarré, ce contrat ne doit plus être modifié par le générateur.
> Contexte : remédiation des 7 GO-gates de l'audit indépendant `docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md`
> en vue d'un nouveau run de 30 jours en démo T212. Ce contrat remplace celui du sprint précédent.

## Critères d'Acceptation Automatisés

### GO-gate 1 — Timeout + idempotence des ordres (F-30)
- [ ] Critère 1 : `safe_request` applique un `timeout` par défaut (10 s) à toutes les requêtes, y compris les GET portefeuille.
- [ ] Critère 2 : Le POST d'ordre market transite par une fonction dédiée avec `timeout=15`.
- [ ] Critère 3 : Sur `RequestException`/timeout d'un POST d'ordre, la position broker est re-vérifiée (`GET /equity/positions`) AVANT tout retry — si la position est apparue (BUY) ou disparu (SELL), aucun re-POST n'émis.
- [ ] Critère 4 : Le retry sur 429/TooManyRequests reste autorisé (ordre rejeté = non exécuté).
- [ ] Critère 5 : Aucun chemin de code ne rejoue un POST d'ordre sans réconciliation intermédiaire.

### GO-gate 2 — Stop-loss mouvant + TP côté broker (F-31)
- [ ] Critère 6 : Le payload d'achat contient `stopLoss` = round(prix×0.90, 2) et `takeProfit` = round(prix×1.08, 2) (prix absolus).
- [ ] Critère 7 : Après fill confirmé, un ordre stop dédié `GOOD_TILL_CANCEL` (quantité négative = vente) existe pour la position ; son id et son prix sont persistés dans `state["active_position"]` (`stop_order_id`, `stop_price`).
- [ ] Critère 8 : Le ratchet ne déplace le stop broker que vers le HAUT (`desired > stop_price + 0.01`), cible `round(peak_price × 0.90, 2)`.
- [ ] Critère 9 : Si le DELETE du stop réussit mais le replacement échoue, un CRITICAL est loggé et un replacement à l'ancien niveau est tenté (jamais de position sans stop connu).
- [ ] Critère 10 : `sync_state_from_t212` réconcilie `stop_order_id`/`stop_price` depuis les ordres actifs broker (state survit aux redémarrages).

### GO-gate 3 — Fill confirmé + prix réel (F-32)
- [ ] Critère 11 : Après 2xx sur un BUY, l'état et la DB ne sont écrits qu'après confirmation de la position broker (poll) ; `entry_price_etf` = `averagePricePaid` broker.
- [ ] Critère 12 : `insert_transaction` (BUY) enregistre le prix de fill réel (`averagePricePaid`), pas le prix signal Yahoo.
- [ ] Critère 13 : Fill non confirmé sous le délai → ERROR loggé, aucun write de state, aucune transaction DB.

### GO-gate 4 — Unité de volatilité + seuils (F-33)
- [ ] Critère 14 : La volatilité passée au decision engine et au weight manager est **quotidienne** (plus de multiplication par √252).
- [ ] Critère 15 : `adaptive_thresholds` = {strong_buy: 0.4375, buy: 0.15, hold_upper: 0.05, hold_lower: -0.05, sell: -0.125, strong_sell: -0.5625} (équivalence 1/0.8 documentée en commentaire).
- [ ] Critère 16 : Un test de régression vérifie que l'amortissement régime (×0.8) n'est PAS appliqué pour une vol quotidienne de 0.02 (ETF calme).

### GO-gate 5 — Données (F-34)
- [ ] Critère 17 : Aucune donnée macro synthétique/aléatoire n'est générée ni mise en cache (le bloc « Method 4 » est supprimé ; échec des sources → None + ERROR log).
- [ ] Critère 18 : Les caches macro (AV_/MULTI_/FRED_PDR_) ne sont servis que si âgés de ≤ 7 jours.
- [ ] Critère 19 : Le fallback cache de prix après échec de téléchargement n'est accepté que si le cache a ≤ 3 jours ; sinon exception → le cycle ne trade pas.

### GO-gate 6 — Scheduler (F-35)
- [ ] Critère 20 : Une seconde instance du scheduler termine immédiatement (CRITICAL + exit non nul) quand `scheduler.lock` est détenu.
- [ ] Critère 21 : Un verrou périmé (> 2 h) est cassé automatiquement.
- [ ] Critère 22 : Une exception non `KeyboardInterrupt` dans l'itération de boucle est loggée et la boucle CONTINUE (le scheduler ne meurt plus).
- [ ] Critère 23 : Le Morning Brief est rattrapé s'il n'a pas encore été généré pour le jour en cours (plus de fenêtre 01:00-01:59 unique).

### GO-gate 7 — Mesure du run (F-36)
- [ ] Critère 24 : Le state par ticker persiste `equity` = initial_budget + P&L réalisé (FIFO sur l'historique broker) + P&L latent.
- [ ] Critère 25 : Le journal CSV écrit une colonne `T212_Equity` alimentée par cette equity (plus le mélange position/cash `T212_Capital`).
- [ ] Critère 26 : En mode T212, `performance_monitor` reçoit le `portfolio_value` = equity réelle (plus la constante 1000 €), `active_positions` et `cash_balance` réels.

### Non-régression
- [ ] Critère 27 : La suite mockée complète passe : 0 échec (196 tests préexistants + nouveaux).
- [ ] Critère 28 : Les invariants existants restent vrais : `write_db = not is_t212`, sentinelle win_rate −1.0, budgets 1000 €/ticker, `CYCLE_TIMEOUT` 40 min, annulation anti-thread-orphelin.
- [ ] Critère 29 : Aucun secret/DB/artefact runtime n'est ajouté à git (`.gitignore` inchangé en couverture).

## Protocole d'Évaluation

* **Commande de tests (Windows)** :
  ```
  .venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp
  ```
  *(ignorer les harness live : test_crawl4ai, check_*, bench_*, run_short_backtest, test_full_cycle)*
* **Sonde broker démo (avec feu vert utilisateur uniquement)** : `uv run python tests/check_t212_stops.py` → valide stopLoss/takeProfit attachés + ordre stop + DELETE sur compte démo.
* **Smoke** : `uv run main.py --ticker SXRV.DE --t212` → 1 cycle complet démo, vérifier dans `trading.log` : ordre (si signal), confirmation fill, stop broker, equity.
* **Comportement attendu** : tests 0 échec ; sonde OK ou fallback documenté ; smoke sans exception non gérée.

*Figé le 2026-08-19 avant la première ligne de code du sprint.*
