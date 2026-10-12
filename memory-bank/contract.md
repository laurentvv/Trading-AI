# Contrat de Validation — Sprint « Migration pullback » (2026-10-12)

> **Rôle** — Contrat de validation technique négocié entre la planification et l'évaluation.
> Liste d'assertions strictes et testables. **Figé juste avant l'écriture de la première ligne de code** :
> une fois le code démarré, ce contrat ne doit plus être modifié par le générateur.
> Contexte : migration complète vers la méthode d'achat sur pullback (décision utilisateur du 2026-10-12),
> plan `docs/PLAN_MIGRATION_PULLBACK_2026-10-12.md`. Ce contrat remplace celui du sprint « GO-gates PROD »
> (2026-08-19, consultable dans l'historique git).

## Critères d'Acceptation Automatisés

### Univers et données (F-43, F-44)
- [ ] Critère 1 : L'univers U1 est construit depuis `GET /equity/metadata/instruments` et ne contient que des instruments `type == "STOCK"` à suffixe `_US_EQ`. Un instrument d'un autre type ou suffixe n'y entre jamais.
- [ ] Critère 2 : La correspondance T212 → Yahoo est déterministe et testée (`AAPL_US_EQ` → `AAPL`, classes d'actions `BRK.B` → `BRK-B`) ; un ticker non résolu est journalisé et exclu, jamais deviné.
- [ ] Critère 3 : Le filtre de liquidité du jour D n'utilise que des données ≤ D (cours ≥ 5 $, volume en dollars moyen 20 j ≥ 20 M$, ≥ 252 séances).
- [ ] Critère 4 : Le stockage journalier en masse est incrémental (seules les barres manquantes sont téléchargées) et refuse une série dont la dernière barre a plus de 3 jours ouvrés (GO-gate 5 conservé).
- [ ] Critère 5 : Aucune donnée synthétique n'est produite pour combler un trou ; une action sans données valides est exclue du jour.

### Règles pullback (F-45)
- [ ] Critère 6 : Le screener retient une action si au moins un rendement à 21, 63 ou 126 séances est ≥ +30 % et si son rang centile sur cet horizon est ≥ 75 (paramètres exposés, valeurs par défaut testées).
- [ ] Critère 7 : La détection de consolidation exige des creux pivots ascendants, ATR(5)/ATR(20) < 0,8, des EMA 9 et 21 de pente ≥ 0, EMA 21 > EMA 50 et une clôture ≥ EMA 21 × 0,98 ; un cas « plus hauts et plus bas descendants sous moyennes descendantes » est rejeté (test dédié).
- [ ] Critère 8 : L'AVWAP est ancré au plus haut de la consolidation et égale Σ(prix typique × volume) / Σ volume depuis l'ancre (test numérique).
- [ ] Critère 9 : Un jour de setup exige un plus bas à ≤ 1,5 % d'au moins un niveau de support et une clôture au-dessus de ce niveau ; le déclencheur vaut le plus haut du jour de setup.
- [ ] Critère 10 : Le stop vaut le plus bas du jour de setup ; un setup dont la distance entrée-stop dépasse 3 % est ignoré.
- [ ] Critère 11 : La taille vaut floor((0,5 % × équité) / (entrée − stop)) au pas de quantité, plafonnée à 25 % de l'équité par ligne ; le risque ouvert total ne dépasse jamais 3 % et l'exposition totale jamais 100 %.
- [ ] Critère 12 : Les sorties vendent 1/3 à 3R et 1/3 à 5R, puis le reste à l'ouverture qui suit une clôture sous l'EMA 9.
- [ ] Critère 13 : Toutes les règles sont des fonctions pures de données ≤ D ; un test prouve qu'ajouter des barres futures ne change pas la décision de D.

### Banc portefeuille (F-46)
- [ ] Critère 14 : Le moteur exécute l'entrée à max(ouverture, déclencheur) et le stop à min(ouverture, stop) ; si l'entrée et le stop tombent le même jour, le stop est supposé touché après l'entrée.
- [ ] Critère 15 : Les frais par défaut sont 0,15 % de change + 10 pb de spread et de glissement par sens ; le PFU de 30 % est appliqué aux plus-values nettes annuelles ; l'équité est tenue en EUR.
- [ ] Critère 16 : Le rapport compare la stratégie au buy & hold Nasdaq-100 EUR, au buy & hold S&P 500 EUR, à la règle MA200, au contrôle « entrées au hasard » et au contrôle « momentum seul ».
- [ ] Critère 17 : Le calibrage n'utilise que 2015-2020 ; la fenêtre 2021 → fin des données n'est utilisée qu'en test (vérifié par un test qui lève une erreur si une variante est calibrée sur la fenêtre de test).
- [ ] Critère 18 : Le rapport publie l'espérance en R, le taux de réussite, les trades par an, le CAGR, le Sharpe, le Sharpe déflaté (nombre de variantes), le drawdown maximal, les frais et l'impôt payés, et les résultats année par année.
- [ ] Critère 19 : Le verdict de la porte du banc (plan §2.1) est calculé par le code et écrit dans le rapport ; aucune phase d'exécution réelle ne démarre sur un verdict négatif.

### Filtres IA (F-47)
- [ ] Critère 20 : Le LLM vision reçoit au plus 15 candidats par jour et renvoie un JSON validé (`score`, `veto`, `analyse` obligatoires) ; une réponse invalide ou absente laisse la décision « règle seule » inchangée.
- [ ] Critère 21 : L'IA ne peut que classer ou mettre un veto : aucun trade n'est créé hors des candidats du screener (test dédié).
- [ ] Critère 22 : Chaque candidat est journalisé avec la décision « règle seule » et la décision « règle + IA » (journal A/B).

### Exécution T212 (F-48, F-49)
- [ ] Critère 23 : Chaque position ouverte a un stop GTC chez le broker, posé après la confirmation de l'exécution ; une position sans stop connu est un événement CRITICAL.
- [ ] Critère 24 : Une vente partielle annule le stop, vend la partie, puis replace le stop sur le reste ; si le replacement échoue, un stop d'urgence est tenté et un CRITICAL est journalisé.
- [ ] Critère 25 : Aucun POST d'ordre n'est rejoué sans réconciliation broker ; « échec de lecture ≠ vide » est conservé pour les positions et l'historique.
- [ ] Critère 26 : Les horaires de séance sont calculés dans le fuseau `America/New_York` (test sur les semaines de décalage de l'heure d'été).

### Non-régression
- [ ] Critère 27 : La suite mockée complète passe (0 échec) à chaque PR.
- [ ] Critère 28 : Les phases 1 à 4 ne modifient ni `src/t212_executor.py` ni `schedule.py` (le scheduler démo actuel continue de tourner).
- [ ] Critère 29 : Aucun secret, base de données ou artefact runtime n'est ajouté à git ; aucun secret n'apparaît dans les journaux.

## Protocole d'Évaluation

* **Commande de tests (Windows)** :
  ```
  .venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp
  ```
  *(ignorer les harness live : test_crawl4ai, check_*, bench_*, run_short_backtest, test_full_cycle)*
* **Banc** : rapport `docs/BACKTEST_PULLBACK_<date>.md` produit par le code du dépôt (aucun chiffre écrit à la main).
* **Sonde broker démo (feu vert utilisateur uniquement)** : ordre stop d'achat, réservation du cash, validité `DAY`, stop de vente GTC après exécution.
* **Comportement attendu** : tests 0 échec ; verdict de porte calculé ; sonde OK ou repli documenté.

*Rédigé le 2026-10-12 ; figé à la validation du plan par l'utilisateur, avant la première ligne de code.*
