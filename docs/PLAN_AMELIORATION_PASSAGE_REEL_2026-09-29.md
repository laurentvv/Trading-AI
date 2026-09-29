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

> **Mise à jour du 2026-09-29 (soir) : le cap a changé.** Après l'étude des stops (`docs/STOPS_ET_VISION_MOYEN_LONG_TERME_2026-09-29.md`), l'utilisateur a tranché : **cœur Nasdaq-100 conservé sans stop broker, poche active de 10 % du capital pilotée par l'ensemble de modèles (LLM et TimesFM au centre du projet), satellite pétrole tactique, décisions hebdomadaires, entrée en une fois, alerte à −35 % depuis le pic.** Les §2, phase 2, phases 3 et 4, §4, §5 et §7 ci-dessous ont été réécrits en conséquence. Les §1, phase 0 et §6 restent l'historique factuel du run démo 2 ; le mode actuel du démo (« legacy », un stop GTC par position) reste régi par le GO-gate 2 tant que la phase 2 n'est pas livrée.

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

### 1.3 Règles de sortie : elles plafonnent les gagnants et retardent la sortie des perdants

`src/t212_executor.py` empile six mécanismes, tous calibrés court terme :

| Règle | Paramètre | Effet pour du moyen/long terme |
|---|---|---|
| Take-profit | +8 % (`TAKE_PROFIT_TARGET`) | Plafonne chaque gain à 8 %. Le TP attaché chez le broker est **rejeté (400) à chaque achat** (3/3 sur la démo) et provoquait un double POST via le repli sur un ordre nu (retiré par la PR #95) ; à confirmer qu'un compte réel ne l'accepte pas davantage. |
| Trailing stop logiciel | −3 % depuis le pic, dès +0,5 % de gain | CRUDP a une volatilité quotidienne de 1,9 à 3,5 % : **moins d'un écart-type d'une séance**, donc déclenchement quasi certain. SXRV (1,1 %) : environ 3 σ, déclenché en quelques semaines. |
| Time-stop | 15 jours calendaires (`MAX_HOLDING_DAYS`) | **Force la vente même si la position est en gain.** Incompatible avec un horizon de plusieurs mois. |
| Garde anti-perte | bloque tout SELL de modèle en dessous de −0,2 % | **18 ventes bloquées.** Elle ne bloque que les ventes **décidées par les modèles** : le hard-stop (−10 %) et le time-stop (jour 15, perte ≤ 5 %) la contournent, donc une position perdante n'est jamais gardée indéfiniment, mais les modèles ne peuvent pas la sortir. Asymétrie inverse du principe « couper les pertes, laisser courir les gains ». La tolérance de 0,2 % est aussi plus étroite que le spread réel (le 24/09 : valeur affichée 1 005,82 €, fill à 1 002,51 €, soit 0,33 %). |
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

**Passer d'un « trader intraday multi-modèles » à un « portefeuille en trois livres » : un cœur passif Nasdaq-100, une poche active de 10 % où l'ensemble de modèles (LLM et TimesFM compris) doit prouver sa valeur, et un satellite pétrole tactique.**

- **Cœur (≈ 90 %)** : Nasdaq-100 acheté en une fois et conservé. **Aucun stop broker, aucune vente décidée par un modèle.** Le filtre MA200 et le momentum ne pilotent pas le cœur : ils servent de **comparateurs** que la poche active doit battre.
- **Poche active (10 % du capital, soit 3 000 €)** : c'est ici que vivent les modèles. L'ensemble décide chaque semaine son niveau d'exposition (0 / 50 / 100 % de la poche). Elle n'est adoptée que si elle bat, hors échantillon et nette de frais et d'impôt, à la fois le buy & hold et la règle MA200. Sinon elle reste passive, et le cœur n'en souffre pas.
- **Satellite pétrole** : tactique, jugé séparément, prélevé **dans** la poche de 10 % (décision de l'utilisateur du 2026-09-30 : le pétrole compte dans les 10 %, pas de budget séparé). Bloqué tant qu'il n'y a pas de source de prix fiable.
- **Cadence hebdomadaire** : une session de décision par semaine. Le reste du temps, seulement des contrôles de santé et de risque, sans LLM.
- **Filet de sécurité** (remplace le stop broker du cœur) : alertes du watchdog, alerte de perte de portefeuille à **−35 %** depuis le pic (alerte puis décision humaine, pas de vente automatique), commande manuelle « tout liquider », plafond de taille de la poche.
- **Validation** : tout changement de la poche passe d'abord par le banc de la phase 1 (walk-forward, net de frais et d'impôt, **contre le buy & hold, la MA200 et un timing aléatoire à rotation égale**). Le démo sert à vérifier que le live reproduit le backtest, pas à découvrir l'edge.

**Ce que la poche peut et ne peut pas faire (arithmétique, pas prévision).** 10 % de 30 000 € = 3 000 €. Si la poche gagne 20 % dans l'année, cela ajoute 600 €, soit +2 % du capital ; si elle perd tout, −10 %. Le rendement du portefeuille vient donc surtout du cœur. La poche est un **budget de risque borné pour tester les modèles en réel**, pas un moteur de performance par construction.

**Risque du cœur à regarder en face.** Tolérance déclarée : environ −30 %. Le Nasdaq-100 a déjà connu des baisses plus profondes (ordres de grandeur, à confirmer : environ −83 % de 2000 à 2002, environ −35 % en 2022). L'alerte à −35 % est donc un événement plausible en marché baissier sévère, pas une rareté. La conduite à tenir quand elle se déclenche doit être **écrite à l'avance** (voir phase 4).

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
6. **Ablation par modèle, test équitable** : chaque voix (TimesFM, classique, LLM texte, LLM vision, oil_bench, council, Grebenkov, HMM, momentum simple) est testée **seule**, puis l'ensemble moins une voix, selon **le même protocole pré-enregistré** (périodes, variantes autorisées et seuils d'adoption écrits avant de lancer). Les LLM posent un problème particulier de fuite d'information (leurs données d'entraînement contiennent le futur du backtest) : voir §3, phase 2.3 pour les parades. **Aucune voix n'est retirée ni déclarée utile sans preuve** ; une voix dont le mauvais résultat vient d'une entrée cassée (sentiment, council, oil_bench) est classée « à réparer », pas « inutile ».

**Livrable :** `docs/BACKTEST_WALKFORWARD_<date>.md` avec le tableau par stratégie et par modèle, plus une décision écrite sur les modèles conservés.

### Phase 2 : refonte « cœur Nasdaq-100 + poche active + satellite pétrole » (3 à 4 semaines ; chaque brique validée par le banc de la phase 1)

Le mode actuel (« legacy » : un stop GTC par position, cycles de 30 min) reste intact et sélectionnable. Le nouveau mode s'active par configuration (`STRATEGY_MODE=core_sleeve`), pour que le démo en cours ne bouge pas avant la bascule.

**Architecture cible**

| Livre | Part du capital | Instrument | Qui décide | Protection |
|---|---|---|---|---|
| **Cœur** | ≈ 90 % (moins une réserve de cash) | Nasdaq-100, obligatoire (SXRV.DE rodé par le démo ; alternatives à comparer, §2.2) | Personne : acheté en une fois, conservé | Pas de stop broker. Alerte −35 % du portefeuille, liquidation manuelle |
| **Poche active** | 10 % (3 000 €) | Nasdaq-100 en exposition 0 / 50 / 100 % de la poche | Ensemble de modèles, chaque semaine | Plafond de taille + coupe-poche (§2.5) |
| **Satellite pétrole** | Dans les 10 % (décision du 2026-09-30) ; répartition avec l'exposition Nasdaq-100 de la poche à fixer par le banc | À trancher (§2.4) | Ensemble pétrole (oil_bench, TimesFM, LLM), chaque semaine | Plafond de taille ; stop très large à évaluer sur le banc |

**2.1 Cadence hebdomadaire**
- **Session hebdomadaire** : snapshot des données après la clôture américaine du vendredi ; pendant le week-end tournent TimesFM, le classique, les LLM (texte, vision, oil_bench) et le council, qui devient la **réunion de la semaine** (rétrospective de la semaine puis vote). Résultat : **un enregistrement de décision figé** (entrées, sorties brutes de chaque voix, fournisseur et modèle LLM réellement utilisés, version des modèles, empreinte du prompt). Exécution **le lundi vers 10 h** (après la première heure de cotation), avec contrôle de fraîcheur des données (GO-gate 5) et de spread.
- **Backtest aligné** : décision à la clôture du vendredi, exécution à l'ouverture du lundi, plus un coût de spread. Même calendrier partout.
- **Jours ouvrés** : tâche légère, **sans LLM ni TimesFM** : santé du scheduler, fraîcheur des données, lecture broker, alerte de drawdown. Les appels LLM tombent de ~17 par jour et par ticker à ~1 par semaine, ce qui supprime le problème des 429 et des 503 en cadence normale.
- Le Morning Brief et FinAcumen quotidiens restent (contexte lu par la session hebdomadaire), mais ne décident rien seuls.

**2.2 Cœur : achat en une fois, conservation, aucune vente logicielle**
- **Instrument** : SXRV.DE est rodé par le démo. Avant le réel, comparer sur des faits vérifiés : frais courants, encours, écart de suivi, spread mesuré sur le compte, devise de cotation (EUR ou USD), disponibilité et fractions chez T212. Les ETF américains ne sont pas accessibles depuis un compte européen (à vérifier), et le compte-titres est le cadre retenu.
- **Entrée** : contrôle des données et du spread, ordre au marché en séance continue (hors première et dernière demi-heure), fill confirmé (GO-gate 3), fill partiel complété le jour même. Un **ordre test de faible montant** sur le même instrument précède le solde, pour vérifier fill, frais, alertes et réconciliation. Le solde part ensuite en une fois (décision de l'utilisateur ; l'étude des stops rappelle qu'un investissement immédiat bat l'étalement dans la majorité des cas historiques).
- **Invariant de code** : un ordre SELL sur un livre de rôle `core` est **refusé** par l'exécuteur, sauf via la commande manuelle `liquidate` (§2.5). C'est une garde de code testée, pas une règle de modèle.
- **Rééquilibrage** : aucun par vente du cœur (chaque vente réalise une plus-value imposable, hypothèse 30 %). Les écarts de poids se corrigent avec de nouveaux versements éventuels.

**2.3 Poche active : moteur, et test équitable des modèles (LLM et TimesFM au centre)**

*Principe.* Chaque voix a sa chance selon les mêmes règles. Aucune n'est retirée sans preuve, aucune n'est créditée sans preuve.

1. **Ré-spécifier chaque voix à l'horizon hebdomadaire** (les défauts constatés en §1.2 sont des défauts de spécification, pas de concept) :
   - **TimesFM** : entrée = série tradée (pas ^NDX si l'ETF est tradé en EUR), horizon de 5 à 13 semaines, décision par les **quantiles** (P(rendement > 0), asymétrie q90/q10) plutôt que médiane contre 0,5 %. Mesurer d'abord son **biais** (prévu moins réalisé) sur cinq ans, puis le corriger.
   - **Classique** : cible à trois classes avec zone neutre, sur 4 à 13 semaines ; variables de tendance et de régime (pente MA200, momentum 3 à 12 mois, volatilité réalisée, drawdown en cours). La cible à 1 jour disparaît.
   - **LLM texte et vision** : prompt hebdomadaire, sortie = niveau d'exposition (0 / 50 / 100 %) + confiance, température fixée, **modèle et fournisseur épinglés** pour la session (voir point 3).
   - **oil_bench** : alimenté uniquement par des données fraîches (garde EIA déjà en place) ; **council** : ses entrées sont réparées avant tout vote (plus de win rate sur 1 trade, plus d'alertes HIGH en boucle).
   - **HMM** : re-spécifié comme détecteur de régime de volatilité (aujourd'hui il discrétise des rendements) ; **TensorTrade** : un modèle par ticker, entraîné sérieusement (≥ 10⁵ pas) ou retiré ; **sentiment** : réparé (cache quotidien sous le quota Alpha Vantage, score au niveau de l'article, proxy de ticker) puis revalidé, ou retiré.
2. **Test équitable, identique pour toutes les voix** : walk-forward strict sans donnée future ; comparateurs = poche 100 % investie, MA200 avec hystérésis, **timing aléatoire à rotation égale** (1 000 tirages, pour savoir si le résultat dépasse la chance) ; métriques nettes de coûts et d'impôt ; nombre de variantes comptabilisé (Sharpe déflaté) ; **protocole pré-enregistré** dans le dépôt avant le premier rejeu.
3. **LLM : trois parades à la fuite d'information.**
   - Rejeux avec **prompts masqués** (ni date, ni nom d'indice, prix normalisés) pour réduire la reconnaissance de période.
   - Évaluation privilégiée sur la **période postérieure à la date de coupure connue** du modèle épinglé.
   - Surtout, un **journal à terme** dès maintenant : chaque session hebdomadaire fige la voix de chaque LLM (le journal d'audit #97 en enregistre déjà onze). C'est la seule preuve réellement hors échantillon. Limite honnête : environ 52 observations par an, donc un pouvoir statistique faible, d'où les revues à 13, 26 et 52 semaines (§2.7).
   - **Épingler le modèle** : la passerelle multi-fournisseurs bascule sur un autre modèle quand le premier échoue, ce qui rend la mesure non reproductible. La session hebdomadaire épingle un fournisseur principal, journalise chaque bascule (`fallback=vrai`) et marque ces votes.
4. **Fusion en exposition de poche** : score pondéré vers des paliers 0 / 50 / 100 %, **hystérésis** (deux sessions consécutives pour changer de palier), détention minimale de 4 semaines. Poids de départ égaux entre les voix retenues, ensuite figés par le walk-forward et revus au plus une fois par semestre (à cadence hebdomadaire, 60 observations par voix représentent plus d'un an : plus de repondération adaptative en live).
5. **Adoption** : la poche n'est active que si elle passe la porte « Edge » de la phase 4. Sinon elle reste passive (Nasdaq-100 ou cash), **mais les voix continuent d'être journalisées et mesurées** : ne pas mesurer reviendrait à les retirer sans preuve.

**2.4 Satellite pétrole**
- **Prérequis bloquant** : une source de prix fiable (CRUDP.PA : flux Yahoo gelé à 82 %). À instruire : autre ETC/ETF pétrole coté et disponible chez T212, ou contrat de référence (CL=F, BZ=F) pour le signal et un ETC pour l'exécution (risque de base et de roll à mesurer). Décision après mesure du roll et du contango sur l'instrument réel.
- **Rôle** : tactique, exposition 0 / 50 / 100 % de sa part, décision hebdomadaire, voix = oil_bench, TimesFM sur la série pétrole, LLM vision et texte. Testé selon le §2.3.
- **Plafond** : le pétrole compte dans les 10 % (décision du 2026-09-30) : poche NDX + pétrole ≤ 10 % du capital, contrôlé par R3. La répartition interne est un paramètre du banc.
- **Protection** : la décision « pas de stop » concerne le cœur. Pour le satellite, très volatil, un **stop très large** est une variante à mesurer avec `scripts/stop_study.py` ; décision à prendre sur le résultat.

**2.5 Couche de risque du mode cœur + poche (remplace le GO-gate 2 pour ce mode)**

| # | Invariant | Vérification |
|---|---|---|
| R1 | **Alerte portefeuille à −35 % depuis le pic** (equity broker, cash compris), alerte Nextcloud, ré-armement à −30 % ; pas de vente automatique. Alertes d'information à −20 % et −30 % (proposition) | Test avec equity simulée ; alerte reçue |
| R2 | **Commande `liquidate`** : vend toutes les positions au marché après confirmation explicite, journalise, arrête le scheduler | Exécutée sur le démo, positions à 0 |
| R3 | **Plafond de poche** : poche + satellite ≤ 10 % de l'equity totale, contrôlé avant chaque ordre | Test unitaire : un ordre qui dépasse est refusé |
| R4 | **Coupe-poche** : si l'equity de la poche tombe sous 50 % de son allocation, elle passe à plat et alerte (proposition, à confirmer) | Test avec equity simulée |
| R5 | **Cœur non vendable par le logiciel** (§2.2) | Test : SELL `core` refusé |
| R6 | **Réconciliation hebdomadaire** état local ↔ broker ; un écart (vente ou achat manuel) est signalé, jamais « corrigé » par un ordre automatique | Test avec broker simulé |
| R7 | **Watchdog conscient du rôle** : « position sans stop » n'est critique que pour un livre déclaré protégé ; le cœur est attendu sans stop | Test du watchdog |
| — | **Restent en vigueur** : GO-gates 1 (idempotence des ordres), 3 (fill confirmé), 4 (volatilité quotidienne), 5 (fraîcheur des données), 6 (verrou du scheduler), 7 (equity FIFO) | Suite existante |

`AGENTS.md` (GO-gate 2) est mis à jour dans la PR qui livre cette couche, pas avant.

**2.6 Découpage en PR (fichiers pressentis, à confirmer à la lecture du code)**

| PR | Contenu | Critère d'acceptation |
|---|---|---|
| 2-a | Modèle « livres » : rôle (`core` / `sleeve` / `oil`), poids cibles à la place de `INITIAL_BUDGETS`, état par livre, `STRATEGY_MODE` | Mode legacy inchangé (suite verte) ; état par livre persisté |
| 2-b | Session hebdomadaire (`--weekly`) + tâche légère quotidienne + enregistrement de décision figé | Un enregistrement complet par session ; 0 appel LLM en semaine |
| 2-c | Exécuteur : garde SELL `core`, plafond de poche, coupe-poche, entrée en une fois | Tests R3 à R5 |
| 2-d | Alerte −35 %, `liquidate`, réconciliation, watchdog conscient du rôle, mise à jour de `AGENTS.md` | Tests R1, R2, R6, R7 |
| 2-e | Voix ré-spécifiées à l'horizon hebdomadaire (TimesFM quantiles, classique 3 classes, prompts LLM, council réparé, HMM) | Chacune évaluable par le banc, entrées fraîches |
| 2-f | Épinglage du modèle LLM + journalisation des bascules | Bascule visible dans le journal |
| 2-g | Fusion en exposition de poche avec hystérésis | Reproduit le backtest sur les mêmes semaines |

**2.7 Revues et règles de décision de la poche (proposition à confirmer)**
- Revues à **13, 26 et 52 semaines** de journal à terme, sur critères écrits à l'avance : écart à la poche passive, à la MA200 et au timing aléatoire, net de coûts.
- Par voix, trois issues : **retenue**, **à réparer** (avec la cause identifiée), **retirée**. Une voix n'est retirée qu'après **deux revues défavorables** appuyées sur des chiffres, jamais sur un seul run court.
- Une voix qui ne se prête pas à un test rétrospectif honnête (LLM) est jugée sur le journal à terme, avec les limites statistiques annoncées.

### Phase 3 : démo de conformité, run 3 (8 à 12 semaines, stratégie gelée)

- **Contrat gelé** (`memory-bank/contract.md`) avant le démarrage. Aucun changement de stratégie pendant le run. Mode `core_sleeve`, cadence hebdomadaire : 8 à 12 sessions seulement.
- Budget démo à l'échelle réelle (30 k€ virtuels si le compte démo le permet), pour valider tailles, quantités et arrondis.
- **Répétitions obligatoires** : entrée en une fois du cœur ; commande `liquidate` ; alerte −35 % déclenchée par une equity injectée ; session du lundi manquée (machine éteinte) puis rattrapage ; coupure du scheduler détectée et alertée sur Nextcloud ; bascule d'un fournisseur LLM journalisée.
- Chaque semaine, **replay du backtest sur les mêmes semaines**. Critère : **écart live/backtest inférieur à 0,5 % par ordre** (implementation shortfall) et expositions identiques.
- Ce run **ne prouve pas l'edge** (8 à 12 sessions, c'est trop peu, et le cœur est passif). Il prouve que **le live reproduit fidèlement le backtest** et que les garde-fous fonctionnent.

### Phase 4 : passage en réel

Pré-requis (tous obligatoires) :

| Porte | Critère |
|---|---|
| Edge (poche active) | Walk-forward net de frais et d'impôt, protocole pré-enregistré : la poche bat **la poche passive et la règle MA200** hors échantillon et dépasse le **timing aléatoire à rotation égale** (Sharpe déflaté pris en compte). **Le cœur, passif par décision, n'a pas d'edge à prouver.** Si la poche échoue, elle reste passive et le passage en réel du cœur n'est pas bloqué |
| Conformité | Phase 3 : écart live/backtest < 0,5 % par ordre, expositions identiques |
| Intégrité | 0 double ordre, **0 vente du cœur non commandée**, invariants R1 à R7 testés, écart DB/broker < 0,5 %, 0 écriture de test en PROD |
| Disponibilité | Watchdog et alertes Nextcloud testés ; 100 % des sessions hebdomadaires tenues en démo |
| Broker live | Types d'ordres, limites de requêtes et instrument (`SXRVd_EQ` ou remplaçant) **vérifiés sur le compte live** par un ordre test de faible montant ; clé API live aux permissions minimales |
| Filet de sécurité | Commande `liquidate` testée ; alerte −35 % testée ; **protocole écrit de décision quand l'alerte tombe** (par exemple : pas d'ordre dans les 48 h, revue avec des critères fixés à l'avance) |
| Fiscalité | Taux du compte-titres confirmé (30 % = hypothèse), impact intégré au banc |

Entrée en réel : ordre test de faible montant sur le même instrument, puis le solde des 30 k€ en une fois (décision de l'utilisateur). Le calendrier reste à la main de l'utilisateur.

---

## 4. À ne pas faire

- **Ajouter des modèles** avant d'avoir le banc de la phase 1 (leçon Kronos, juillet 2026 : implémenté, puis rejeté au backtest).
- **Retoucher seuils et poids sur quelques semaines de live** (ADR-002 a calé les poids sur 4 semaines de marché baissier, puis le marché a monté).
- Juger une voix sur le « win rate » d'une poignée de trades, ou laisser le council voter sur ces métriques.
- **Retirer ou promouvoir un LLM ou TimesFM sur un seul backtest ou moins de 26 sessions hebdomadaires** ; changer de fournisseur LLM sans le journaliser.
- **Poser un stop ou une règle de vente sur le cœur** (décision de l'utilisateur). Toute exception passe par lui.
- Passer en réel en changeant simplement `T212_ENV=live` : les types d'ordres et les limites live ne sont pas vérifiés par un ordre réel.

---

## 5. Ordre d'exécution recommandé

1. **Fait** : phase 0 (hygiène et mesure) et phase 1, tranche A (références).
2. **Ensuite** : phase 1, tranche B (rejeu équitable des voix en walk-forward, ablation, décision écrite). Prérequis : source de prix du pétrole, spread mesuré, comparaison des instruments Nasdaq-100.
3. **Phase 2** : PR 2-a à 2-g (§2.6), chacune validée par le banc. Les PR 2-a à 2-d ne dépendent pas de la tranche B.
4. **Phase 3** : démo de conformité (8 à 12 semaines, stratégie gelée).
5. **Phase 4** : passage en réel si les portes sont vertes.

Tant que la phase 2 n'est pas livrée, le run démo 2 peut continuer pour roder l'exploitation, mais **ses résultats de P&L ne doivent pas servir à la décision GO/NO-GO**.

---

## 6. Avancement de la phase 0 (mis à jour le 2026-09-29)

Le démo a été **arrêté le 29/09 à 21h25** pour permettre les corrections (position SXRV.DE ouverte, protégée par son stop GTC
broker #55701244265 @ 1 386,81 ; ce stop n'est plus remonté tant que le scheduler est à l'arrêt). Chaque correction est une PR séparée.

| Point | Statut | PR |
|---|---|---|
| 0.1 Isolation des tests (CWD temporaire par test) | fait | #93 |
| 0.2 Purge des 8 fausses lignes de `trading_history.db` | fait en local (sauvegarde `.bak-2026-09-29`, base non versionnée) | n/a |
| 0.3 Séparation PROD/DEV | runbook rédigé, **à exécuter par le propriétaire** (choix du tag et du dossier) | #92 (`RUNBOOK_EXPLOITATION.md`) |
| 0.4 FinAcumen de nouveau exécuté après chaque brief | fait | #94 |
| 0.5 Watchdog + alertes + relance | fait ; **canal d'alerte et installation de la tâche à faire par le propriétaire** | #98 |
| 0.6 Budget d'appels T212 | fait : régulateur par endpoint selon les limites officielles ; 4 lectures espacées de 5,4 à 5,5 s, toutes 200 sur la démo | #96 |
| 0.7 Retrait du `takeProfit` attaché | fait | #95 |
| 0.8 Journal d'audit complet | fait : 11 voix, consensus, désaccord, issue réelle de l'exécution ; migration testée sur les 477 lignes réelles | #97 |
| 0.9 Sources mortes | **diagnostiqué, volontairement non modifié** (voir ci-dessous) | n/a |
| 0.10 Doc T212 en réel + sonde broker | doc vérifiée (voir ci-dessous) ; **sonde démo `check_t212_stops.py` non lancée (accord requis)** | #96 (`TRADING212_API_GUIDE.md`) |

### 0.9 : pourquoi le modèle « sentiment » vaut 0,00 à chaque cycle (diagnostic)

Deux causes cumulées, toutes deux confirmées :

1. **Quota Alpha Vantage** : offre gratuite = 25 requêtes par jour. Le pipeline en émet environ 76 par jour (2 requêtes × 2 tickers × ~19 cycles).
   Après les premiers cycles, l'API répond `Information` (limite atteinte, constaté en direct le 29/09) et le code retombe sur Google News RSS,
   dont le score est codé en dur à `0.0`. Les « 10 headlines, score 0,00 » du log sont ce repli.
2. **Filtre de ticker** (`news_fetcher.py`, « correctif Bug C » de juillet) : seules les lignes `ticker_sentiment` dont le ticker vaut `SXRV.DE`
   ou `CRUDP.PA` sont retenues, or Alpha Vantage ne connaît pas ces tickers Yahoo. Même avec du quota, le compteur reste à 0 et le score à 0.

**Décision : ne pas le réparer en phase 0.** Le réactiver remettrait en jeu un votant à **0,16** (le plus gros poids) dont le score Alpha Vantage
est réputé biaisé à la hausse, sans backtest possible (pas d'historique de sentiment) : ce serait un changement de stratégie déguisé en correctif.
Il est traité en phase 1 (ablation) : soit retiré, soit réparé (cache quotidien pour rester sous 25 requêtes, score au niveau article, proxy de ticker)
puis revalidé. En attendant, il vote HOLD 0,50 sans effet sur le score (les HOLD ne comptent pas au numérateur).

### 0.10 : ce que dit la documentation officielle Trading 212 (vérifié le 2026-09-29)

- **Ordres limit, stop et stop-limit disponibles par l'API en réel depuis le 29/01/2026** (annonce du staff sur le fil « Trading 212 API Update » ;
  l'article d'aide « Trading 212 API key » le confirme). La phrase « seuls les ordres au marché sont exécutables en réel », encore visible dans
  d'anciennes versions de la doc, est périmée. Le risque M9 du §1.4 est donc **levé sur sources écrites** ; il reste à le confirmer par **un premier
  stop réel de faible montant** avant toute montée en charge.
- API toujours en bêta, comptes Invest et Stocks ISA uniquement, clés API démo et réelles distinctes.
- Limites de débit par compte et par endpoint : voir `TRADING212_API_GUIDE.md` §5. `GET /equity/orders` : **1 requête / 5 s**, cause des 429 observés.

---

## 7. Les piliers du système de décision (mis à jour le 2026-09-29 soir)

Un pilier est ce qui doit rester même si tout le reste est retiré. Critères : **mesurable hors échantillon** (à terme pour les LLM), **explicable**,
**utile au capital de 30 k€**. Le banc de la phase 1 tranche pour la **poche active** ; il ne remet pas en cause le cœur, qui est passif par décision.

| # | Pilier | Composants | Rôle | Pourquoi |
|---|---|---|---|---|
| 1 | **Cœur Nasdaq-100 et comparateurs** | Cœur conservé sans stop ; MA200, momentum 3-12 mois et buy & hold comme références | Porte l'essentiel du rendement ; les comparateurs définissent ce que la poche doit battre | Sur SXRV.DE, aucune règle simple n'a battu le buy & hold sur le Sharpe (§8) ; la MA200 réduit le drawdown au prix de rendement. |
| 2 | **Couche de risque et d'exécution** | GO-gates 1, 3 à 7 ; en mode cœur + poche, invariants R1 à R7 (alerte −35 %, `liquidate`, plafond de poche, cœur non vendable), watchdog, régulateur d'appels | Protège le capital **même quand les modèles se trompent** | Le vrai acquis du projet. Le stop broker du cœur est remplacé par un filet d'alertes et de commandes manuelles, selon la décision de l'utilisateur. |
| 3 | **Ensemble de modèles de la poche active** | **TimesFM 3.0** (quantiles) et **modèle classique** (cibles hebdomadaires), plus Grebenkov et HMM re-spécifié | Décident l'exposition de la poche et du satellite | Cœur du projet. Les défauts constatés en §1.2 (série non tradée, cible à 1 jour) sont des défauts de spécification à corriger, puis à mesurer avec le protocole équitable de la phase 2.3. |
| 4 | **Voix qualitatives, LLM** | LLM texte et vision, oil_bench, council, FinAcumen, morning brief | **Voix votantes à part entière** de la poche et du satellite pétrole, avec un modèle épinglé et un journal à terme | Cœur du projet. Non backtestables honnêtement (fuite d'information) : jugés sur un journal à terme, revues à 13, 26 et 52 semaines. Leur droit de vote n'est retiré qu'après deux revues défavorables chiffrées. |
| 5 | **Banc de mesure** | Backtest walk-forward, timing aléatoire de contrôle, journal d'audit (#97), enregistrement figé de chaque session hebdomadaire | Rend chaque réglage vérifiable | Sans lui, les poids ont été calés sur 4 semaines de marché baissier (ADR-002). |

**À réparer avant de les juger** : sentiment (quota et filtre de ticker, voir §6), council (entrées sur micro-échantillons), TensorTrade (2 000 pas, un seul modèle pour deux tickers), oil_bench (EIA périmé). **Désactivé** : Vincent Ganne. Ces défauts sont des bugs d'entrée ; ils ne disent rien de la valeur du concept.

**Ordre de grandeur du problème constaté** : 0,54 des 0,95 de poids nominal (57 %) était porté par sentiment (0,16), TimesFM (0,15), classique (0,13) et council (0,10), dont deux morts (sentiment, council sur du bruit) et deux mal spécifiés (TimesFM sur une autre série, classique à 1 jour).

**Conséquence sur l'univers** : pour le pétrole, la source de prix et le coût de roll passent avant tout (§2.4).

Ce document est une analyse d'ingénierie, pas un conseil en investissement personnalisé.

### Constat complémentaire : fenêtre sans stop lors d'une vente non exécutée

Ce constat concerne le **mode legacy** (un stop par position). Dans `_execute_sell_order`, le stop broker est annulé **avant** l'envoi de la vente. Si l'ordre au marché est accepté mais ne s'exécute pas, la position peut rester sans protection jusqu'au cycle suivant. La PR #96 (en revue) reformule l'alerte et reprotège quand la vente n'est pas confirmée. Il devient **sans objet pour le cœur** (pas de stop) ; il reste à traiter pour un éventuel stop du satellite.


## 8. Avancement de la phase 1, tranche A : les références (mis à jour le 2026-09-29)

Moteur de backtest, métriques, bootstrap et rapport livrés dans la PR `feat/backtest-baselines` (`src/backtest/`, `docs/BACKTEST_BASELINES_2026-09-29.md`). Premiers résultats, à lire avec leurs limites (4 ans, un seul cycle) :

- **CRUDP.PA n'est pas backtestable** : flux Yahoo gelé à 82 % (vivant depuis le 2026-01-08, 184 séances). Il faut une autre source de prix pour l'instrument pétrole, ou un autre instrument. Le contrat CL=F n'est qu'un proxy non ajusté du roll.
- **SXRV.DE, 2022-07 → 2026-09** : aucune règle simple ne bat le buy & hold sur le Sharpe (1,12). La MA200 avec hystérésis 2 % offre un drawdown de −15 % contre −26,7 % pour 6 points de CAGR en moins.
- **Les règles de sortie actuelles** (TP +8 %, trailing −3 %, time-stop 15 j), avec une entrée toujours haussière, tombent à 8,6 % de CAGR contre 21,8 % : à 0 pb de coût elles font 20,0 %, l'essentiel de la perte vient du **churn** (83 trades, rotation ×40 par an) et non de la seule troncature des gains.
- **À faire ensuite (tranche B)** : rejeu des modèles de l'ensemble en walk-forward, ablation par modèle, décision écrite sur les modèles conservés. Prérequis : trancher la source de prix du pétrole et mesurer le spread réel (par côté ou aller-retour) sur le compte.

## 9. Décisions et réécriture de la phase 2 (2026-09-29, soir)

Décisions de l'utilisateur intégrées : aucun stop broker sur le cœur ; poche active de 10 % ; Nasdaq-100 obligatoire ; entrée en une fois ; satellite pétrole tactique ; cadence hebdomadaire ; tolérance de baisse du cœur ≈ −30 % ; alerte à −35 % ; compte-titres ; alertes Nextcloud Talk ; **LLM et TimesFM au cœur du projet**.

À confirmer par l'utilisateur (propositions, pas décisions) : coupe-poche à −50 % de l'allocation ; alertes d'information à −20 % et −30 % ; revues à 13/26/52 semaines et règle « deux revues défavorables » ; jour et heure de la session hebdomadaire (vendredi soir → lundi 10 h) ; ordre test avant l'entrée en une fois ; stop très large sur le satellite pétrole.
