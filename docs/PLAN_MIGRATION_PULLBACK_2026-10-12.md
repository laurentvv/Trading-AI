# Plan de migration : méthode « achat sur pullback » (2026-10-12)

> **Décision de l'utilisateur du 2026-10-12 : migration complète du projet** vers la méthode d'achat sur pullback
> décrite par Martin Luk (fil @OtherSideBJ du 2026-10-12). Univers : tout ce qui est facilement disponible sur
> Trading 212, avec une grosse partie sélection, suivi et choix. Déclencheur d'entrée : **journalier**. LLM et
> TimesFM : placés là où ils servent le mieux la nouvelle méthode.
>
> Ce plan **remplace** `docs/PLAN_AMELIORATION_PASSAGE_REEL_2026-09-29.md` (cœur Nasdaq-100 à 90 % sans stop,
> poche active de 10 %, cadence hebdomadaire, satellite pétrole). Il ne touche pas au scheduler démo en cours :
> l'ancien pipeline tourne jusqu'à la bascule (phase 5).

---

## 0. Ce que l'on sait de la méthode, et ce que l'on ne sait pas

- **La source n'est pas une preuve.** Le rendement de 969,8 % est celui du gagnant d'un concours (US Investing
  Championship 2025, catégorie actions) sur un an : sur des milliers de participants, le meilleur résultat dit peu
  de l'espérance d'une règle. C'est l'effet survivant signalé par la réponse du fil : un test sur plus de 70 000
  pullbacks en actions chinoises donne environ 45 % de trades gagnants, et le gain vient surtout de la coupe
  rapide des pertes.
- **La méthode est discrétionnaire** (lecture de graphiques, thèmes « populaires », déclencheur sur bougie de
  1 à 15 min). Ce plan la traduit en **règles mécaniques journalières**. Le banc mesure cette traduction, pas
  Martin Luk.
- **Comme toute règle de ce projet, elle doit battre le buy & hold du Nasdaq-100**, nette de frais et
  d'impôt, hors échantillon. C'est le coût d'opportunité réel : l'alternative passive reste disponible
  à tout moment. Si elle échoue au banc (phase 3), on ne la passe pas en réel.

## 1. Les quatre étapes traduites en règles

Paramètres par défaut, puis **variantes pré-enregistrées** (les seules autorisées au calibrage, pour limiter le
surapprentissage ; le nombre total de variantes alimente le Sharpe déflaté du banc).

### 1.1 Univers (« tout ce qui est facilement disponible sur T212 »)

- Source : `GET /equity/metadata/instruments` (relevé du 2026-10-12, compte démo) : 18 473 instruments, dont
  **12 168 actions** : 6 898 en USD (`XXX_US_EQ`), 3 140 en EUR, 1 401 en GBX, 542 en CAD, 134 en CHF.
- **U1 (phases 1 à 6) : actions US `_US_EQ`.** La correspondance avec Yahoo est directe (`AAPL_US_EQ` → `AAPL`,
  avec les cas `BRK.B` → `BRK-B` à traiter). C'est le vivier de la méthode (momentum, thèmes, liquidité).
- U2 (après la démo, optionnel) : actions européennes via correspondance ISIN → symbole Yahoo (pas de frais de
  change en EUR). Hors périmètre de ce sprint.
- Filtre de liquidité, recalculé chaque jour : cours ≥ 5 $ (variante 3 $), volume en dollars moyen sur 20 jours
  ≥ 20 M$ (variante 10 M$), au moins 252 séances d'historique. Instruments non `STOCK` exclus.

### 1.2 Étape 1 : sélection des actions fortes (screener)

- Rendements à 21, 63 et 126 séances (1, 3 et 6 mois).
- Retenue si **au moins un horizon ≥ +30 %** et rang centile ≥ 75 sur cet horizon dans l'univers liquide du jour
  (top 25 % ; variantes 70 et 80).
- **Thèmes** : nombre d'actions fortes par industrie (champ industrie Yahoo, mis en cache). Dans la version de
  base, ce compte sert de bonus de classement. Variante : une industrie doit compter au moins 3 actions fortes.

### 1.3 Étape 1 (suite) : consolidation qui se renforce

Calculée sur les 10 à 30 dernières séances après le plus haut récent :

- **Plus bas ascendants** : les deux derniers creux pivots (fractal à 2 barres) montent.
- **Volatilité qui se resserre** : ATR(5) / ATR(20) < 0,8 (variante 0,7).
- **Moyennes** : EMA 9 et EMA 21 plates ou montantes (pente sur 5 séances ≥ 0), EMA 21 > EMA 50, clôture
  ≥ EMA 21 × 0,98.
- **Exclusion** : plus hauts et plus bas descendants avec clôture sous des moyennes descendantes (le cas « hors
  critère » de la source).

### 1.4 Étape 2 : zone de support et signal

- Niveaux : EMA 9, 21 et 50 ; **AVWAP ancré au plus haut de la consolidation** ; anciens plus hauts pivots ;
  gaps non comblés.
- Le jour D est un **jour de setup** si son plus bas touche un niveau à 1,5 % près et si la clôture repart au-dessus
  (rebond). Score de confluence = nombre de niveaux dans la zone (≥ 2 = zone forte).
- **Déclencheur journalier** : ordre d'achat stop au **plus haut de D** (+ 1 tick), valable la séance D+1
  (variante : 3 séances, ré-ancré chaque jour). Si D+1 ouvre au-dessus du déclencheur, l'achat se fait à
  l'ouverture. Si D+1 casse le plus bas de D avant le déclenchement, le setup est annulé (« le prix traverse le
  support sans rebond, on attend »).

### 1.5 Étape 3 : stop et taille, avant l'ordre

- **Stop** = plus bas de D. Distance d = (entrée − stop) / entrée. **Trade ignoré si d > 3 %** (variante 4 %). En
  journalier, cette règle sélectionne naturellement les jours à faible amplitude, ce qui rejoint la contraction
  de volatilité.
- **Taille** = (0,5 % de l'équité) / (entrée − stop) actions, avec plafond de 25 % de l'équité par ligne, pas de
  levier (exposition totale ≤ 100 %), **risque ouvert total ≤ 3 %** de l'équité (environ 6 lignes). Avec un stop
  à 3 %, une ligne pèse environ 17 % du compte.
- Pas d'entrée dans les 5 séances avant une publication de résultats, si la date est connue (un gap de résultats
  traverse le stop et fait perdre bien plus que 1R).

### 1.6 Étape 4 : sorties

- **1/3 vendu à 3R, 1/3 à 5R** (R = entrée − stop initial).
- Reste : **clôture sous l'EMA 9 → vente à l'ouverture suivante**.
- Variantes : stop remonté au prix d'entrée après 3R ; allègement sur extension (clôture > EMA 9 + 2 × ATR) ;
  filtre de marché (« si le marché est fort ») : entrées autorisées seulement si le Nasdaq-100 clôture au-dessus
  de son EMA 21.
- Le stop initial reste en place chez le broker jusqu'à la sortie (GO-gate 2 redevient la règle pour **toutes**
  les positions).

## 2. Le banc (phase 3) : la porte avant tout code d'exécution

Le banc actuel (`src/backtest`) rejoue une exposition 0..1 sur un seul actif. La méthode demande un **moteur de
portefeuille multi-actions** :

- **Événementiel journalier, sans regard vers le futur** : les décisions de D utilisent les données ≤ clôture D.
  Exécution de D+1 sur OHLC : entrée à max(ouverture, déclencheur) si le plus haut ≥ déclencheur ; stop exécuté à
  min(ouverture, stop) si le plus bas ≤ stop (le gap baissier est payé en entier). **Si l'entrée et le stop
  tombent le même jour, on suppose le pire** (entrée puis stop). Les objectifs 3R/5R sont exécutés au niveau
  exact, ou à l'ouverture si elle les dépasse.
- **Frais** : change T212 de 0,15 % par sens sur les actions en USD, plus 10 points de base de spread et de
  glissement par sens (variantes 5 et 25). Équité tenue en EUR (EURUSD quotidien). **PFU de 30 %** sur les
  plus-values nettes de l'année.
- **Références** : buy & hold Nasdaq-100 en EUR (SXRV.DE), buy & hold S&P 500 en EUR, règle MA200 sur le
  Nasdaq-100.
- **Contrôles** : (a) entrées au hasard dans le même univers momentum, avec les mêmes stops et sorties. Ce
  contrôle teste si les étapes 1.3 et 1.4 ajoutent quelque chose. (b) « momentum seul » : étape 1.2 et sorties,
  sans setup.
- **Biais du survivant** : la liste T212 et Yahoo ne contiennent que les actions cotées aujourd'hui (les faillites
  et rachats ont disparu). Le résultat sur U1 est donc **optimiste**. Pour le mesurer, le même banc est rejoué sur
  un **univers à la date** (composition historique du S&P 500 et du Nasdaq-100, source publique à choisir avec
  l'utilisateur). Les résultats sont aussi publiés année par année.
- **Walk-forward** : calibrage des variantes sur 2015-2020, **test sur 2021 → 2026-10** (2022 baissier inclus),
  sans aucun réglage sur la fenêtre de test. Le Sharpe déflaté et le bootstrap apparié sont déjà dans
  `src/backtest/metrics.py`.
- **Sortie** : `docs/BACKTEST_PULLBACK_<date>.md` avec espérance en R par trade, taux de réussite, gain et perte
  moyens en R, trades par an, exposition moyenne, CAGR, Sharpe, drawdown maximal, frais et impôt payés.

### 2.1 Portes de passage proposées (à valider par l'utilisateur avec ce plan)

| Porte | Condition |
|---|---|
| **Banc → construction de l'exécution** | Sur la fenêtre de test, nette de frais et d'impôt : CAGR **et** Sharpe > buy & hold Nasdaq-100 EUR ; drawdown maximal ≤ 35 % (seuil d'alerte décidé le 2026-09-29) ; espérance ≥ +0,25 R par trade sur au moins 150 trades ; contrôle « entrées au hasard » battu (bootstrap, p < 0,05) ; résultat encore positif sur l'univers à la date. |
| Échec au banc | **Pas de passage en réel.** On peut seulement explorer les variantes pré-enregistrées (une seule fois, comptées dans le Sharpe déflaté), sinon retour à l'option passive. |
| **Démo → réel** | Au moins 8 semaines **et** 40 trades en démo, avec un glissement et un taux d'exécution des ordres stop compatibles avec les hypothèses du banc (R moyen réalisé dans l'intervalle de confiance du banc). |
| **Montée en charge** | Réel à 0,25 % de risque par trade pendant 4 semaines, puis 0,5 %. |

## 3. Rôle des LLM et de TimesFM dans la nouvelle méthode

La règle mécanique **propose** ; l'IA **classe et met son veto**, elle ne crée jamais un trade.

| Brique | Nouveau rôle | Mesure |
|---|---|---|
| **LLM vision** (`get_visual_llm_decision`) | Le « chartiste » de l'étape 1 (« ouvrir les graphiques un par un ») : pour les 15 meilleurs setups du jour, graphique hebdo + journalier (EMA 9/21/50, AVWAP, niveaux) → JSON `{score 0-10, plus_bas_ascendants, contraction, qualite_support, veto, analyse}`. | Démo, en avant : un LLM a déjà vu le passé, un backtest de son filtre serait truqué. |
| **LLM texte + recherche web** (`web_researcher`, morning brief) | Thèmes et catalyseurs : résume les thèmes chauds à partir des industries du set momentum et des actualités ; veto sur catalyseur négatif (augmentation de capital, révision des prévisions, résultats imminents). | Démo, en avant. |
| **TimesFM 3.0** | Variable de classement : médiane de prévision à 10 séances et écart des quantiles rapportés au R du setup. | Banc, **seulement sur la période postérieure à ses données d'entraînement** ; démo ensuite. |
| **Weekend Council** | Revue hebdomadaire des trades (R réalisés, respect des règles, glissement par rapport au plan). Plus de vote. | Qualitatif. |
| **Morning brief** | Contexte de marché (régime) et actualité de la nuit sur les positions ouvertes. | Qualitatif. |

**Protocole A/B** : chaque candidat est journalisé avec la décision « règle seule » et la décision « règle + IA ».
Le filtre IA n'est gardé que s'il améliore l'espérance en R sur au moins 100 candidats journalisés.

**Retirés du chemin d'exécution** (le code reste jusqu'à la phase 6, pour permettre un retour arrière) :
classic, grebenkov, oil_bench et client EIA, hmm, tensortrade, vincent_ganne, sentiment, gestionnaire de poids
adaptatifs et moteur de décision d'ensemble sur ETF.

## 4. Exécution sur Trading 212 (phases 4 et 5, seulement après la porte du banc)

- **Plan nocturne** (après la clôture US) : mise à jour des données, screener, setups, classement IA, plan
  d'ordres (déclencheur, stop, quantité, niveaux 3R et 5R) écrit dans `data_cache/pullback_plan_<date>.json`,
  puis résumé dans Nextcloud Talk.
- **Entrées** : ordres stop d'achat (`POST /equity/orders/stop`, quantité positive) valables la séance, posés à
  l'ouverture US. **À vérifier par une sonde démo, avec le feu vert de l'utilisateur** : achat stop accepté,
  réservation du cash, validité `DAY`, exécution sur gap. Repli : déclenchement logiciel par la boucle de suivi.
- **Après exécution** (confirmée chez le broker, GO-gate 3) : ordre stop de vente GTC au stop du plan (GO-gate 2).
  La fenêtre sans stop entre l'exécution et la pose du stop est au plus d'un intervalle de suivi (5 min). Ce risque
  est accepté et documenté.
- **Ventes partielles à 3R et 5R** : côté logiciel, à chaque passage de la boucle de suivi. On annule le stop, on
  vend la partie, puis on replace le stop sur le reste : c'est le schéma de `_execute_sell_order`, car un stop
  réserve les titres et empêche un ordre limite de vente sur les mêmes titres.
- **Suivi à l'EMA 9** : évalué après la clôture ; vente au marché à l'ouverture suivante.
- **Portefeuille multi-positions** : l'équité vient du compte (T212 summary), pas de budgets par ticker
  (`INITIAL_BUDGETS` disparaît) ; P&L FIFO par position ; journal par trade avec R.
- **Scheduler** : séance US de 15:30 à 22:00, heure de Paris, calculée dans le fuseau `America/New_York` (les
  passages à l'heure d'été ne tombent pas le même jour en Europe et aux États-Unis) ; boucle de suivi toutes les
  5 min ; plan nocturne ; morning brief ; council hebdomadaire. Verrou, watchdog et alertes inchangés.
- **Débit API** : environ 20 POST stop ou limite par jour au maximum, très en dessous de la limite (1 toutes les
  2 s). `GET /equity/orders` (1 toutes les 5 s) est regroupé par passage.

## 5. Phases et PR

| Phase | Contenu | PR |
|---|---|---|
| 0 | Ce plan, contrat gelé, feature list, mémoire du projet | celle-ci |
| 1 | Univers T212 → Yahoo (U1), filtre de liquidité ; stockage journalier en masse (Parquet, mise à jour incrémentale) | 1 |
| 2 | Règles pullback en fonctions pures (screener, consolidation, support, déclencheur, risque, sorties) + tests | 1 |
| 3 | Moteur de portefeuille multi-actions, références, contrôles, univers à la date, rapport → **porte** | 1 à 2 |
| 4 | Filtres IA (LLM vision, thèmes et catalyseurs, TimesFM) + journal A/B | 1 |
| 5 | Exécuteur T212 multi-positions + nouveau scheduler + sonde démo des ordres stop d'achat | 2 |
| 6 | Retrait de l'ancien ensemble du dépôt, documentation (README, AGENTS.md, guide API) | 1 |
| 7 | Démo de conformité (≥ 8 semaines, ≥ 40 trades), puis réel progressif | — |

Les phases 1 à 3 ne touchent ni au scheduler démo ni à `src/t212_executor.py`.

## 6. Ce qui change dans les décisions et les invariants

- **Remplacé** : cœur Nasdaq-100 à 90 % sans stop, poche active de 10 %, cadence hebdomadaire, satellite pétrole
  (2026-09-29 et 2026-09-30). La PR #106 (livres cœur/poche, `STRATEGY_MODE`) devient sans objet. Les
  modifications « swing » non commitées du 2026-10-10 aussi.
- **Conservé** : GO-gates 1 (ordres non rejoués à l'aveugle), 2 (stop broker sur chaque position, de nouveau sans
  exception), 3 (fill confirmé), 5 (pas de données synthétiques, données périmées refusées), 6 (verrou du
  scheduler), 7 (équité FIFO) ; `write_db = not is_t212` ; « échec de lecture ≠ vide » ; régulation du débit ;
  aucun secret dans git ni dans les journaux ; alerte de portefeuille à −35 % depuis le pic.
- **Sans objet ou à réécrire** : GO-gate 4 (seuils de vote sur la volatilité quotidienne des ETF), budgets par
  ticker, liste `TICKERS` fixe, horaires du scheduler.

## 7. Risques et limites

- **Biais du survivant** dans l'univers actuel (mesuré par l'univers à la date, sans être annulé).
- **Le déclencheur intraday de la source n'est pas reproduit** : le journalier entre plus tard et avec un stop plus
  large, donc avec un R moins bon que celui décrit.
- **Gaps et glissement** : une perte peut dépasser 1R (gap de résultats, nouvelle de nuit). Le banc le paie en
  entier.
- **Frais** : environ 0,4 à 0,5 % par aller-retour (change et spread) pour un stop de 2 à 3 %, soit 15 à 20 % de
  chaque R.
- **Concentration** : environ 6 lignes de 15 à 25 % du compte, toutes en actions US momentum, avec le risque de
  change EURUSD en plus.
- **Fiscalité** : beaucoup de trades par an sur un compte-titres chez un courtier étranger, donc une déclaration à
  préparer (taux de 30 % toujours à confirmer).
- **Filtre IA** : son apport ne se mesure qu'en avant, donc lentement.

## 8. Questions ouvertes (non bloquantes pour les phases 1 à 3)

1. Capital au démarrage réel (30 k€ selon l'objectif) et risque par trade (0,5 % comme la source ?).
2. Source publique de la composition historique S&P 500 et Nasdaq-100 pour le contrôle du biais du survivant
   (téléchargement à valider au moment de la phase 3).
3. Actions européennes (U2) après la démo ?
4. Données intraday payantes plus tard, pour un déclencheur plus fin ?

---

## 9. Résultat du banc (phase 3, 2026-10-12) : la porte n'est PAS franchie

Rapport complet, produit par le code : `docs/BACKTEST_PULLBACK_2026-10-12.md` (6 595 symboles, calibrage 2015-2020, test 2021 -> 2026-10-12, net de frais et d'impôt).

- **Test hors échantillon** : Pullback (variante retenue `ema9_apres_3R`) **-22,6 %/an**, Sharpe -1,93, drawdown -78,9 %, 680 trades, réussite 19 %, espérance -0,55 R. Buy & hold Nasdaq-100 EUR : +14,6 %/an, Sharpe 0,74.
- **Les 6 variantes pré-enregistrées perdent toutes**, au calibrage comme au test.
- **La stratégie fait moins bien que le hasard** : entrées au hasard dans le même univers momentum, mêmes stops et sorties : médiane -15,0 %/an, espérance -0,45 R ; la stratégie est battue par 95-98 % des tirages. Les étapes consolidation / support / AVWAP n'ajoutent donc rien dans cette traduction journalière.
- **Ce n'est pas un artefact d'hypothèses** : avec l'ordre intrajournalier le plus favorable (entrée annulée si le stop cède d'abord, cibles avant stop) la stratégie reste négative (-7,8 %/an, espérance -0,16 R) ; sans aucun frais, -14,1 %/an dans l'hypothèse du plan. Le biais du survivant joue en faveur de la stratégie : le vrai résultat ne serait pas meilleur.
- **Lecture** : un stop de 2 à 3 % (plus bas de la veille) est plus étroit que l'amplitude journalière normale d'un titre momentum ; il se fait toucher dans la majorité des cas (réussite de 19 à 30 %, soit environ le 1/4 d'une marche aléatoire avec cible à 3R), et les frais de 50 pb par aller-retour en prennent environ 0,2 R par trade.
- **Conséquence selon le plan (§2.1)** : pas de construction d'exécution (phases 4 à 7 suspendues). Les variantes pré-enregistrées sont épuisées. Ce banc ne teste pas l'intraday de la source (bougies de 1 à 15 min) : la méthode d'origine reste non testée, pas réfutée. Décision à prendre par l'utilisateur.

