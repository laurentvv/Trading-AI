# Plan d'amélioration : passage en réel (30 k€, horizon moyen/long terme)

**Date :** 2026-09-29. **Statut :** proposition, aucun code modifié.
**Base factuelle :** run démo T212 n° 2 (03/09 → 29/09/2026). Sources : `trading_journal.csv` (477 lignes), `trading.log`, `scheduler.log`, `trading_history.db`, `model_performance.db`, `performance_monitor.db`, `t212_portfolio_state.json`, caches parquet et code source.
**Objectif utilisateur :** gagner réellement de l'argent sur le moyen/long terme avec **30 k€ de capital réel de départ**.

> Ce document est un plan d'ingénierie et de validation du logiciel, pas un conseil en investissement personnalisé. La décision d'engager du capital réel, son montant et son calendrier appartiennent à l'utilisateur. La fiscalité est à valider avec un professionnel.

---

## 0. Verdict

**NO-GO réel en l'état.** Ce n'est pas un problème de réglage fin. Trois causes structurelles expliquent les résultats décevants :

1. **Le système est construit pour le court terme alors que l'objectif est le moyen/long terme.** Il tourne en cycles de 30 min, le modèle classique vise la direction à 1 jour, le take-profit est plafonné à +8 %, le trailing se déclenche à −3 % et un time-stop force la vente au bout de 15 jours, même en gain.
2. **Aucun avantage (edge) n'a été démontré.** Le comité de modèles est dominé par des votes biaisés à la baisse ou inactifs. Les poids ont été calés sur 4 semaines de données (ADR-002). Un run démo de 30 jours n'apporte que 15 jours de décision, ce qui est statistiquement inexploitable.
3. **La mesure du démo elle-même est faussée.** Les tests écrivent dans la base de trading du démo. Les modèles sont entraînés sur une autre série que celle qui est tradée. Le scheduler n'a tourné que 66 % du temps.

La barre à battre est connue. Sur 5 ans (09/2021 → 09/2026), le buy & hold fait **SXRV.DE +113 % (drawdown max −31 %)** et **CRUDP.PA +152 % (drawdown max −25 %)**. Pour justifier son existence, un système moyen/long terme doit **capter l'essentiel de cette hausse avec un drawdown plus faible**, net de frais et d'impôts. Sinon, le buy & hold l'emporte par défaut.

---

## 1. Constat chiffré du run démo 2

### 1.1 Résultat vs marché

| | Système (equity T212) | Buy & hold sur la même période (02/09 → 29/09) |
|---|---|---|
| SXRV.DE (1 000 €) | 1 002,92 € (réalisé FIFO +2,33 €, position ouverte le 29/09 @1 534,80) | **+6,1 %** |
| CRUDP.PA (1 000 €) | 995,11 € (aller-retour 15/09 → 24/09 : −4,89 €, −0,49 %) | **+9,8 %** |
| **Total 2 000 €** | **1 998 € (−0,1 %)** | **≈ +160 €** |

- **3 transactions réelles en 26 jours** : une vente SXRV le 08/09 (−3,24 €), un aller-retour CRUDP, un achat SXRV le 29/09. Le critère « ≥ 20 allers-retours » de `PLAN_RUN_DEMO_30J.md` §5 est hors d'atteinte, comme au run 1.
- **Signaux finaux** : CRUDP.PA **1 BUY sur 244 cycles** (129 HOLD, 114 SELL) ; SXRV.DE 25 BUY sur 233.
- Le 24/09, CRUDP a été vendu avec une **confiance de 0,19**. Il est ensuite monté jusqu'à 15,44 (clôture du 29/09 : 15,10).

### 1.2 Comportement de chaque modèle de décision

| Modèle | Poids de base | Comportement observé | Diagnostic |
|---|---|---|---|
| **classic** (RF/GB/LR) | 0,13 | CRUDP : **SELL 244/244**. SXRV : SELL 84, HOLD 149, **0 BUY** | CV F1 du 08/09 : RF **0,23 ± 0,36**, GB 0,35, LR 0,48, donc pire que le hasard. La cible (`features.py:203`) est « rendement du lendemain > 0,1 σ ». La classe 0 regroupe donc les jours plats **et** les jours baissiers, et elle est lue comme SELL, ce qui crée un biais baissier structurel. L'horizon de 1 jour est du bruit. |
| **timesfm** (3.0) | 0,15 | SELL dans **93 % (CRUDP) et 84 % (SXRV)** des cycles, sur un marché haussier | Il prévoit **^NDX / CL=F** (série d'analyse), pas l'ETF tradé. Signal = médiane à 5 j contre un seuil de 0,5 %. Après un SELL TimesFM sur CRUDP, le rendement à 5 j a été de **+6,0 %** en moyenne (n = 7). Mal orienté, mais sur un échantillon minuscule. Les quantiles sont calculés mais inutilisés. |
| **sentiment** | **0,16** (le plus gros poids) | **Score 0,00 sur 158/158 cycles**, donc HOLD 0,50 en permanence | Modèle **mort** : les titres sont récupérés mais le score est nul. Son poids est perdu (les HOLD ne votent pas). Déjà signalé dans `TODO.md` (« 0 headlines »). |
| **llm_text** | 0,12 | SXRV : HOLD 0,95 dans 89 % des cycles. CRUDP : HOLD 80 % | Surtout un vote d'abstention. Impossible à backtester proprement (fuite d'information via les données d'entraînement et le web). |
| **llm_visual** | 0,16 | CRUDP : BUY 62 %. SXRV : BUY 30 % | Seul vote haussier régulier sur le pétrole. Coût : fallbacks gemini_free en 503 (94 échecs) et nvidia en timeout. |
| **tensortrade** (PPO) | 0,04 | Réparti BUY/HOLD/SELL presque au hasard | **2 000 timesteps** au total, le fine-tuning est commenté (`tensortrade_model.py:214`) et **un seul `ppo_model.zip` sert aux deux tickers**. En pratique, c'est une politique quasi aléatoire. |
| **grebenkov** (tendance) | 0,05 | BUY 25 jours sur 27 | **Seul suiveur de tendance, et il avait raison** sur la période. Poids marginal. Absent du journal. |
| **hmm_model** | 0,04 | BUY 13, SELL 13 | Pile ou face. Absent du journal. |
| **oil_bench** (LLM) | 0,08 | BUY 59, SELL 19 | Alimenté par des données EIA périmées (crude_imports vieux de 115 j, refusé 79 fois). Absent du journal. |
| **vincent_ganne** | 0,02 | N/A sur 100 % des cycles | Désactivé (volontairement) mais toujours dans la liste des poids. |
| **council** (week-end) | **0,10** | SELL/HOLD, 36/36 votes | Il raisonne sur des métriques de bruit (« win rate 0 % » calculé sur **1 trade**, alertes HIGH en boucle) et a recommandé la « quarantaine » sur ce bruit. Absent du journal. |
| **FinAcumen** | n/a (morning brief) | **Ne tourne plus depuis le 25/09** | **Régression du commit `752ecb8`** : le bloc FinAcumen de `schedule.py::run_morning_brief` est désormais dans le `except subprocess.TimeoutExpired`, donc il ne s'exécute plus que si le brief dépasse 30 min. |

**Poids adaptatifs** : 12 à 15 observations par modèle et par ticker, en dessous du seuil de 20 (`WIN_RATE_MIN_SAMPLES`). Le mécanisme adaptatif n'a donc **jamais été actif** pendant le run : les poids statiques ADR-002 (calés sur 4 semaines de juin) ont tout décidé.

**Bilan** : les modèles les plus lourds (sentiment 0,16, timesfm 0,15, classic 0,13, council 0,10) sont morts ou biaisés à la baisse. Ceux qui étaient alignés sur la tendance (grebenkov, llm_visual sur le pétrole) pèsent peu. Le consensus ne peut pas produire de BUY sur CRUDP.

### 1.3 Règles de sortie : elles coupent les gagnants et gardent les perdants

`src/t212_executor.py` empile six mécanismes, tous calibrés court terme :

| Règle | Paramètre | Effet pour du moyen/long terme |
|---|---|---|
| Take-profit | +8 % (`TAKE_PROFIT_TARGET`) | Plafonne chaque gain à 8 %. Le TP attaché chez le broker est **rejeté (400) à chaque achat** (3/3) : c'est du code mort qui provoque un double POST. |
| Trailing stop logiciel | −3 % depuis le pic, dès +0,5 % de gain | CRUDP a une volatilité quotidienne de 1,9 à 3,5 % : **moins d'un écart-type d'une séance**, donc déclenchement quasi certain. SXRV (1,1 %) : environ 3 σ, déclenché en quelques semaines. |
| Time-stop | 15 jours calendaires (`MAX_HOLDING_DAYS`) | **Force la vente même si la position est en gain.** Incompatible avec un horizon de plusieurs mois. |
| Garde anti-perte | bloque tout SELL de modèle en dessous de −0,2 % | **18 ventes bloquées.** Les perdants sont conservés jusqu'au hard-stop −10 %. Asymétrie inverse du principe « couper les pertes, laisser courir les gains ». La tolérance de 0,2 % est aussi plus étroite que le spread réel (le 24/09 : valeur affichée 1 005,82 €, fill à 1 002,51 €, soit 0,33 %). |
| Hard-stop | −10 % | Cohérent comme stop catastrophe. |
| Stop broker à cliquet | pic × 0,90 | Bonne protection, même si la machine tombe. Ce stop doit **rester le socle**. |

À quoi s'ajoutent un anti-churn de **4 h** seulement (`MIN_HOLDING_HOURS`) et **17 décisions par jour et par ticker**, alors que les modèles travaillent sur des barres quotidiennes : le journal montre les mêmes votes répétés d'un cycle à l'autre.

### 1.4 Problèmes de mesure et d'exploitation

| # | Problème | Preuve |
|---|---|---|
| M1 | **Les tests écrivent dans la base du démo** | 8 lignes `BUY 10 @100.00 / 2026-09-07 10:00:00` dans `trading_history.db` (ids 1-4, 6, 9-11), une par exécution de la suite. Source : `tests/test_prod_fixes_2026_08_24.py::TestMaxAvailableSizing` appelle `_execute_buy_order` sans neutraliser `insert_transaction`, et `src/database.py` a un `DB_PATH` relatif. **Le démo tourne dans le checkout de dev (`C:\GIT\Trading-AI`)**, pas dans un répertoire PROD séparé. |
| M2 | Disponibilité de **66 %** (477 lignes sur 722 attendues) | Trou du 15/09 15:34 au 22/09 00:07 (**6,4 jours**, 4 séances perdues), 12 h le 09/09, 5 h le 25/09. Le superviseur `.bat` n'a pas relancé et aucune alerte n'a été émise. Critère ≥ 95 % non atteint. |
| M3 | Série analysée ≠ série tradée | SXRV (EUR) est analysé sur ^NDX (USD) : +4,2 % contre +6,1 % sur la période (effet EUR/USD). CRUDP (ETC en EUR, avec roll) est analysé sur CL=F : **+3,0 % contre +9,8 %**. Les modèles prédisent une autre série que celle qui fait le P&L. |
| M4 | 429 sur `/equity/orders` à **chaque cycle** depuis le 29/09 14:34 | « Stop fetch error — local stop state preserved » (16 fois) : le cliquet travaille à l'aveugle. Il y a trop d'appels T212 par cycle (sync ×2, positions, cash, orders, history). |
| M5 | Journal incomplet | Il manque grebenkov, hmm, oil_bench, council, les poids, le score pondéré, l'action exécutée et le motif de sortie. On ne peut pas auditer une décision a posteriori. |
| M6 | Monitoring de perf non significatif | `daily_performance` : 0 ligne. `model_performance_summary` : 0. 58 alertes « win rate critically low » calculées sur 1 trade, qui alimentent ensuite le council. |
| M7 | Échelle non représentative | Démo : 2 × 1 000 €, en tout-ou-rien (`sizing_ratio = 1.0` en dur). Cible : 30 k€. |
| M8 | Sources de données mortes ou bruitées | EIA crude_imports périmé (115 j), gemini_free en 503 (94 fois), crawls OPEC/IEA en échec, avertissement `HF_TOKEN`, `findfont` ×80. |
| M9 | La sonde broker n'a jamais été lancée | `tests/check_t212_stops.py` est toujours marqué « feu vert requis » dans `progress.md`. Le support des ordres stop sur l'**API live** n'est **pas vérifié** (historiquement, l'API live T212 n'acceptait que les ordres au marché ; à confirmer dans la doc officielle actuelle). Si c'est encore le cas, le GO-gate 2 tombe en réel. |

---

## 2. Principe directeur de la refonte

**Passer d'un « trader intraday multi-modèles » à un « portefeuille moyen terme avec filtre de régime ».**

- **Cœur** : exposition longue par défaut sur les actifs retenus, avec une taille calée sur la volatilité.
- **Filtre de régime** : les modèles ne servent plus à entrer et sortir chaque jour. Ils disent si l'on est en régime « risk-on » (exposition pleine), « neutre » (exposition partielle) ou « risk-off » (exposition réduite ou cash).
- **Sorties** : pilotées par la rupture de tendance ou de régime et par un stop catastrophe chez le broker, **quel que soit le P&L latent**. Plus de plafond de gain ni de sortie au calendrier.
- **Cadence** : **une décision par jour** (après la clôture ou avant l'ouverture). Pendant la séance, on ne fait que des contrôles de risque légers (stops, cohérence broker).
- **Validation** : tout changement passe d'abord par un **backtest walk-forward sur 5 ans**, net de frais et d'impôts, **contre le buy & hold et contre des règles simples** (par exemple un filtre MA200). Le démo sert à vérifier que le live reproduit le backtest, pas à découvrir l'edge.

---

## 3. Plan par phases

### Phase 0 : hygiène et mesure fiable (immédiat, 2 à 3 jours, sans changer la stratégie)

| ID | Action | Fichiers | Critère d'acceptation |
|---|---|---|---|
| 0.1 | **Isoler les tests** : fixture `autouse` dans `tests/conftest.py` qui redirige `database.DB_PATH`, `model_performance.db`, `performance_monitor.db`, `t212_portfolio_state.json` et `trading_journal.csv` vers `tmp_path`. Ajouter un garde-fou qui fait échouer la suite si un test ouvre un chemin en dehors de tmp. | `tests/conftest.py`, `src/database.py` (chemin surchargeable par variable d'environnement) | Après `pytest`, `trading_history.db` est identique octet pour octet |
| 0.2 | **Purger les 8 fausses lignes** (après sauvegarde `.bak`) | script ponctuel | La DB ne contient que les fills broker |
| 0.3 | **Séparer PROD et DEV** : répertoire PROD dédié (clone ou worktree sur un tag), déploiement par `git pull --ff-only` d'un tag validé. On ne lance plus jamais pytest dans le répertoire PROD. | `start_scheduler.bat`, `docs/PLAN_MIGRATION_*` | Le scheduler tourne hors du checkout de dev |
| 0.4 | **Réparer FinAcumen** (bloc sorti du `except TimeoutExpired`) + test de non-régression | `schedule.py` | « Lancement de l'analyse profonde FinAcumen » réapparaît chaque jour |
| 0.5 | **Watchdog externe** : tâche Planificateur Windows toutes les 15 min qui relance le scheduler s'il est mort, plus une **notification** (Telegram, ntfy ou mail) si aucun cycle depuis 2 h en séance, position sans stop, 429 persistants ou exception fatale | nouveau script + tâche Windows | Un kill simulé est détecté et relancé en moins de 15 min, avec une alerte reçue |
| 0.6 | **Budget d'appels T212** : un seul snapshot positions, cash et orders par cycle, réutilisé partout ; backoff par endpoint | `src/t212_executor.py` | 0 « Stop fetch error » sur une journée |
| 0.7 | **Supprimer l'attache `takeProfit`** (toujours rejetée, 400) | `src/t212_executor.py` | Un seul POST par achat |
| 0.8 | **Journal complet** : les 11 voix, les poids effectifs, le score pondéré, le signal brut, le signal ajusté du risque, l'action exécutée et le motif (TP, trailing, time-stop, hard-stop, garde) | `main.py`, `src/enhanced_trading_example.py` | Toute décision est reconstructible depuis le CSV |
| 0.9 | Neutraliser les sources mortes : sentiment (0,00), EIA crude_imports (115 j), vincent_ganne ; retirer leur poids plutôt que de les laisser voter HOLD | `src/config_weights.py` | Journal sans modèle constant |
| 0.10 | **Sonde broker** `tests/check_t212_stops.py` en démo, puis **vérification écrite** dans la doc officielle T212 des ordres disponibles sur le compte **live** (stop, stop-limit, GTC) et des limites de requêtes live | `TRADING212_API_GUIDE.md` | Tableau « démo vs live » consigné et signé |

### Phase 1 : banc de vérité, le backtest walk-forward (1 à 2 semaines)

C'est **le livrable qui conditionne tout le reste.** Sans lui, chaque réglage est une supposition.

1. **Moteur** : il rejoue **le vrai code de décision** (`EnhancedDecisionEngine` et les règles de sortie de l'exécuteur, factorisées en fonctions pures) sur les barres quotidiennes 2021 → 2026. Réentraînement walk-forward (fenêtre glissante, **aucune donnée future**). Il remplace ou étend `backtest_prod.py` et `scripts/backtest_ensemble_10y.py`.
2. **Coûts réalistes** : spread et slippage mesurés en démo (0,2 à 0,35 % observés), frais de change le cas échéant, **impôt sur les plus-values réalisées** en option (un trading fréquent réalise des gains imposables chaque année, contrairement au buy & hold, qui les diffère).
3. **Séries correctes** : P&L calculé sur **les séries EUR tradées** (SXRV.DE, CRUDP.PA), et tester l'entraînement des modèles directement sur ces séries plutôt que sur ^NDX et CL=F.
4. **Références obligatoires** : buy & hold ; filtre de tendance simple (au-dessus ou en dessous de la MA200) ; 50 % investi fixe. Une stratégie qui ne bat pas la MA200 simple ne justifie pas sa complexité.
5. **Métriques** : CAGR, Sharpe, Sortino, **drawdown max**, Calmar, temps investi, turnover, nombre de trades, gain moyen/perte moyenne. Robustesse : bootstrap des rendements et **Sharpe déflaté** (on a testé beaucoup de variantes).
6. **Ablation par modèle** : chaque modèle seul, puis l'ensemble moins un modèle. **On ne garde que les modèles qui ajoutent de la valeur hors échantillon.** Les LLM (texte, vision, oil_bench, council) ne sont pas backtestables honnêtement : ils passent en **vote fantôme** (loggé, non décisionnel) jusqu'à preuve sur un échantillon live suffisant.

**Livrable :** `docs/BACKTEST_WALKFORWARD_<date>.md` avec le tableau par stratégie et par modèle, plus une décision écrite sur les modèles conservés.

### Phase 2 : refonte de la stratégie moyen terme (2 à 3 semaines, chaque point validé par le banc de la phase 1)

**2.1 Cadence**
- Une décision de stratégie **par jour** (par exemple 17:45 après la clôture, exécution à l'ouverture suivante ou le lendemain 09:15). Les cycles de 30 min ne servent plus qu'aux contrôles de risque (stops, sync broker, alertes), sans LLM. Appels LLM divisés par environ 17, beaucoup moins de 429 et de 503, et plus de bruit intraday.

**2.2 Cibles et modèles**
- **classic** : horizon de 20 à 60 jours, cible à 3 classes (hausse / zone neutre / baisse) ou régression du rendement futur, avec des variables de tendance et de régime (pente de la MA200, momentum 3 à 12 mois, volatilité réalisée, drawdown en cours). Supprimer la cible à 1 jour.
- **timesfm** : entrée = série tradée, horizon de 20 jours, décision fondée sur les **quantiles** (P(rendement > 0), rapport q90/q10) plutôt que sur la médiane contre 0,5 %. Mesurer d'abord son **biais moyen** (rendement prévu moins réalisé) sur 5 ans, et le corriger ou l'écarter.
- **grebenkov et un momentum simple** : candidats naturels pour le **filtre de régime** (à confirmer par l'ablation).
- **tensortrade** : poids à 0 tant qu'il n'y a pas un modèle par ticker entraîné sérieusement (au moins 10⁵ timesteps, walk-forward) et validé. Sinon, le retirer.
- **hmm** : uniquement comme détecteur de régime de volatilité, s'il est validé. Sinon, poids 0.
- **sentiment et vincent_ganne** : retirés (ou réparés puis revalidés).
- **council** : **consultatif uniquement** (rapport humain), plus de vote tant qu'il est alimenté par des métriques de micro-échantillons.
- **Poids adaptatifs** : gelés sur les poids issus du walk-forward. Pas de repondération en live en dessous de 60 observations par modèle et par ticker.

**2.3 Règles de sortie cohérentes avec le moyen terme**
- **Supprimer le take-profit fixe à +8 %.**
- **Supprimer le time-stop de 15 jours**, ou le remplacer par une sortie « tendance cassée ET aucun progrès depuis N semaines ».
- **Supprimer la garde anti-perte** sur les sorties de régime : une sortie est décidée par une règle, pas par le P&L latent.
- **Trailing** : remplacer le −3 % par un stop fondé sur l'ATR (par exemple 3 × ATR(20)) ou par une sortie « clôture sous MA100/MA200 ». Paramètres choisis par le backtest, pas à la main.
- **Stop catastrophe chez le broker** : conservé (cliquet), avec un niveau calé sur la volatilité de chaque actif (CRUDP est environ 3 fois plus volatil que SXRV).
- **Hystérésis** : seuils d'entrée et de sortie distincts, et **détention minimale de 5 à 10 séances** hors stop catastrophe, pour supprimer le churn.

**2.4 Taille des positions et allocation (dimensionnée pour 30 k€)**
- Abandonner le tout-ou-rien (`sizing_ratio = 1.0` en dur) : exposition par paliers (0 / 25 / 50 / 75 / 100 %) selon le régime et le niveau de conviction.
- **Ciblage de volatilité** par actif : CRUDP, bien plus volatil et soumis au roll, reçoit une part plus faible que SXRV à risque égal.
- Plafond par actif, réserve de cash, et **coupe-circuit portefeuille** (drawdown global supérieur à X % : exposition réduite et alerte).
- `INITIAL_BUDGETS` et la logique d'equity passent en **allocation de portefeuille** (poids cibles), et non plus en « 1 000 € par ticker ».

**2.5 Univers**
- Réévaluer CRUDP.PA : matière première en contango et backwardation, pas de rendement de portage durable, divergence avec CL=F. Le garder seulement si le backtest net de roll montre un apport au portefeuille (décorrélation). Sinon, le remplacer par un actif mieux adapté au moyen terme, à évaluer dans le banc.

### Phase 3 : démo de conformité, run 3 (8 à 12 semaines, stratégie gelée)

- **Contrat gelé** (`memory-bank/contract.md`) avant le démarrage. Aucun changement de stratégie pendant le run.
- Budget démo à l'échelle réelle (allocation cible sur 30 k€ virtuels si le compte démo le permet), pour valider tailles, quantités et arrondis.
- Chaque soir, **replay du backtest sur les mêmes jours**. Critère : **écart live/backtest inférieur à 0,5 % par trade** (implementation shortfall) et signaux identiques.
- Ce run **ne prouve pas l'edge** : 12 semaines sont trop courtes. Il prouve que **le live reproduit fidèlement le backtest** qui, lui, a prouvé l'edge.

### Phase 4 : passage en réel progressif

Pré-requis (tous obligatoires) :

| Porte | Critère |
|---|---|
| Edge | Walk-forward 5 ans net de frais et d'impôts : **drawdown max nettement inférieur au buy & hold** avec un CAGR proche, OU Sharpe > B&H + 0,2 ; meilleur que la règle MA200 simple ; robuste au bootstrap |
| Conformité | Phase 3 : écart live/backtest < 0,5 % par trade, 100 % des signaux reproduits |
| Intégrité | 0 double ordre, 0 position sans stop broker, écart DB/broker < 0,5 %, 0 écriture de test en PROD |
| Disponibilité | ≥ 98 % des décisions quotidiennes prises à l'heure, watchdog et alertes testés |
| Broker live | Types d'ordres, limites de requêtes et instruments (`SXRVd_EQ`, `OD7Fd_EQ` ou remplaçant) **vérifiés sur le compte live** ; clé API live aux permissions minimales |
| Coupe-circuit | Commande manuelle « tout liquider et stopper » testée ; perte journalière et drawdown max du portefeuille déclenchent un arrêt automatique |

Montée en charge : démarrer le réel sur **une fraction** du capital prévu, comparer au démo et au backtest pendant plusieurs semaines, puis augmenter par paliers si les écarts restent dans les tolérances. Le rythme et les montants sont une décision de l'utilisateur.

---

## 4. À ne pas faire

- **Ajouter des modèles** avant d'avoir le banc de la phase 1 (leçon Kronos, juillet 2026 : implémenté, puis rejeté au backtest).
- **Retoucher seuils et poids sur quelques semaines de live** (ADR-002 a calé les poids sur 4 semaines de marché baissier, puis le marché a monté).
- Juger la stratégie sur le « win rate » d'une poignée de trades, ou laisser le council voter sur ces métriques.
- Passer en réel en changeant simplement `T212_ENV=live` : la sonde live et les types d'ordres ne sont pas vérifiés.

---

## 5. Ordre d'exécution recommandé

1. **Cette semaine** : phase 0 (0.1 → 0.10). Tout est non stratégique et sans risque pour le run en cours, sauf 0.3 (déplacer la PROD), à faire un soir hors séance.
2. **Semaines 2-3** : phase 1 (banc walk-forward) et rapport de décision sur les modèles.
3. **Semaines 4-6** : phase 2 (refonte), chaque changement validé sur le banc.
4. **Semaines 7-18** : phase 3 (démo de conformité, stratégie gelée).
5. **Ensuite** : phase 4, si toutes les portes sont vertes.

Tant que la phase 2 n'est pas livrée, le run démo 2 peut continuer pour roder l'exploitation (phase 0), mais **ses résultats de P&L ne doivent pas servir à la décision GO/NO-GO**.
