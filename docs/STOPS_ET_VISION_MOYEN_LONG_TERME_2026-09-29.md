# Stops, ventes, achats et vision moyen/long terme : recherche et options

Document de décision. Il répond à ta remarque du 2026-09-29 : *« un stop sur des positions long et moyen terme, ce n'est pas ma vision »*. Il rassemble (1) ce que dit la recherche, (2) ce que mesure **notre propre banc** sur SXRV.DE et sur le pétrole, (3) les options de conception pour l'achat, la vente et le risque, (4) les décisions que toi seul peux prendre.

Statut : **rien de ce qui suit n'est implémenté.** Les invariants actuels (stop broker sur chaque position, cliquet peak×0,90, `AGENTS.md` GO-gate 2) restent en vigueur jusqu'à ta décision.

> Limite de méthode : les résumés de la recherche viennent de recherches web (résumés d'articles, pages d'éditeurs), pas d'une lecture intégrale des papiers. Les chiffres cités sont ceux des résumés ; à vérifier dans la source avant de fonder un choix sur un chiffre précis.

---

## 1. En une page

1. **Un stop n'est pas neutre.** En marché sans tendance (marche aléatoire), un stop 0/1 *réduit* le rendement espéré ; il n'aide que s'il y a de la persistance (momentum). Le cadre théorique est celui de Kaminski & Lo. Un stop serré sur un actif qu'on veut garder des années est donc, par construction, un pari contre ta propre vision.
2. **Ce qui a de la preuve sur le long terme n'est pas un stop de prix mais une règle de tendance lente** (moyenne mobile 10 mois ≈ MA200, momentum 12 mois) : Faber, Moskowitz-Ooi-Pedersen, Hurst-Ooi-Pedersen (67 marchés, 1880-2016). Elle réduit surtout le **drawdown**, moins le rendement. Mais Zakamulin montre que les performances de ces règles sont souvent gonflées par un biais de anticipation et peu robustes hors échantillon.
3. **Notre banc confirme le profil.** Sur SXRV.DE (2022-07 → 2026-09) : buy & hold 21,8 %/an, drawdown −26,7 % ; MA200 avec hystérésis 15,9 %/an, drawdown −15,0 %. Aucun écart de Sharpe n'est statistiquement significatif. **Les stops fixes (−10 à −30 % depuis l'entrée) ne se déclenchent presque jamais** (une fois à −10 %, jamais au-delà) (le drawdown de −26,7 % est mesuré depuis le sommet, pas depuis l'entrée). Les stops suiveurs donnent des résultats **non monotones** (coûts 25 pb par côté ; −15 % : 25,9 %/an ; −10 % : 19,1 % ; −20 % : 19,4 % ; −25 % : 17,5 %) : c'est du bruit de paramétrage, pas un effet.
4. **Le vrai ennemi n'est pas la vente, c'est la rotation.** Les règles actuelles (TP +8 %, trailing −3 %, time-stop 15 j) font 8,6 %/an contre 21,8 % pour un actif qu'on ne vend jamais ; à 0 pb de coût elles feraient 20,0 %. Tout l'écart vient de 83 trades et d'une rotation de ×40/an.
5. **Le pétrole (CRUDP.PA) est un cas à part** : flux de prix inutilisable (82 % gelé) et, structurellement, un ETC sur contrats à terme subit le coût du roll en contango (jusqu'à 82 % du temps entre 2006 et 2017 selon la littérature). Sur un horizon long, ce n'est pas un actif « à garder » comme un indice actions.
6. **Recommandation de travail (à valider par toi)** : une architecture à deux niveaux, *cœur* investi durablement sans stop de prix + *surcouche de régime* lente et optionnelle, plus un *filet de sécurité opérationnel* (pas un stop de trading). Détail au §5. Toutes les variantes doivent passer le banc avant d'être retenues.

---

## 2. Ce que dit la recherche

### 2.1 Stops de perte (stop-loss)

- **Kaminski & Lo, *When do stop-loss rules stop losses?* (Journal of Financial Markets, 2014).** Sous l'hypothèse de marche aléatoire, un stop 0/1 simple *diminue toujours* le rendement espéré ; en présence de momentum, il peut ajouter de la valeur. Sur des rendements mensuels 1950-2004, certaines règles ajoutent 50 à 100 points de base par mois pendant les périodes de sortie, et à fréquence plus lente elles peuvent augmenter le rendement espéré tout en réduisant nettement la volatilité. Lecture : **l'efficacité d'un stop dépend du régime de marché**, pas d'une vertu propre.
- **Autres études citées par la littérature de synthèse** : certaines trouvent que les stops ne dégradent pas la performance et réduisent les pertes, d'autres qu'ils réduisent le rendement en échange de moins de risque. Conclusion prudente de ces résumés : **effet dépendant des conditions de marché**, ni universellement bon ni universellement mauvais.
- **Mécanique broker (Trading 212).** Un stop devient un ordre au marché une fois déclenché ; le seuil est comparé au dernier prix négocié. Il ne garantit **pas** le prix d'exécution : en cas de gap, l'exécution se fait au prix suivant disponible, parfois très en dessous. Un stop-limit évite le mauvais prix mais peut ne pas s'exécuter du tout. Sur compte réel, l'API permet les ordres limit, stop et stop-limit ; l'API n'est ouverte qu'aux comptes Invest et Stocks ISA ; maximum 50 ordres en attente par ticker. Conséquence : **le stop broker est une assurance contre l'indisponibilité de notre système, pas contre le risque de gap.**

### 2.2 Tendance et momentum (la « vente » à horizon long)

- **Faber, *A Quantitative Approach to Tactical Asset Allocation* (2007).** Règle : investi quand le cours mensuel est au-dessus de sa moyenne mobile simple à 10 mois (≈ MA200), sinon hors marché. Résultat rapporté : rendements de type actions avec volatilité et drawdown de type obligations, sur plusieurs classes d'actifs. Au-dessous de la MA10 mois, le rendement moyen est nettement plus faible et la volatilité plus élevée.
- **Moskowitz, Ooi & Pedersen, *Time series momentum* (JFE, 2012).** Le rendement des 12 derniers mois prédit le rendement du mois suivant ; acheter les actifs à rendement 12 mois positif et vendre les autres rapporte un rendement ajusté du risque significatif, positif sur chacun des 58 contrats liquides étudiés.
- **Hurst, Ooi & Pedersen, *A Century of Evidence on Trend-Following Investing* (JPM, 2017).** 67 marchés, 1880-2016 : rendements positifs, faible corrélation avec actions/obligations à chaque décennie, bonne tenue dans 8 des 10 plus grandes crises d'un portefeuille 60/40.
- **Daniel & Moskowitz, *Momentum Crashes*.** Le momentum est asymétrique négativement : de rares séquences de pertes fortes, surtout après une baisse de marché en régime de forte volatilité (« rebonds »). Une gestion dynamique du momentum double à peu près le Sharpe. Lecture : **une règle de tendance a un risque de « fouet » lors des retournements violents.**
- **Contrepoint : Zakamulin, *Market Timing with Moving Averages*.** Les performances « trop belles » de ces règles viennent en partie d'un biais d'anticipation dans les simulations ; en test hors échantillon, sur 300 formes de pondération et quatre indices, l'avantage sur la stratégie passive n'est faiblement visible que pour le S&P 500, et pas significatif sur la seconde moitié de l'échantillon. **Notre moteur exécute à l'ouverture *t+1* précisément pour éviter ce biais**, mais le point de fond reste : ne pas sur-interpréter un backtest.

### 2.3 Coût de la sortie du marché

- **Bessembinder.** Quatre actions sur sept ont un rendement buy & hold inférieur au bon du Trésor ; 4 % des sociétés expliquent le gain net du marché depuis 1926. Une part importante du rendement excédentaire se concentre sur peu de jours. Lecture double : (a) **détenir un indice large plutôt que quelques titres** protège de ce risque de concentration ; (b) **être dehors les mauvais jours coûte cher** : toute règle de sortie doit être jugée sur le rendement manqué autant que sur la perte évitée.
- **Vanguard, investissement en une fois vs lissé (DCA).** Sur 12 mois, l'investissement en une fois bat le lissage environ 68 % du temps (62 à 74 % selon les pays), pour un gain moyen d'environ 1,2 à 2,4 points selon la part d'actions, parce que les marchés montent plus souvent qu'ils ne baissent. Le lissage ne réduit que le **regret** et la variance d'entrée, il coûte en moyenne du rendement.

### 2.4 Taille des positions plutôt que sortie

- **Moreira & Muir, *Volatility-Managed Portfolios* (JF, 2017).** Réduire l'exposition quand la volatilité est haute (sans la changer quand elle est basse) augmente le Sharpe : +0,15 en moyenne sur les indices de 20 pays OCDE (rapporté), et de 50 à 100 % du Sharpe d'origine pour plusieurs facteurs. Lecture : **on peut réduire le risque par la taille, sans stop de prix.**

### 2.5 Pétrole et matières premières

- **Erb & Harvey** : le rendement d'un contrat à terme se décompose en rendement spot et **rendement de roll**. En contango, le roll coûte chaque mois. 1994-2005 : les futures pétrole ont fait mieux que le spot ; 2006-2017 : moins bien, avec un marché en contango jusqu'à 82 % du temps. Lecture : un ETC pétrole a une **dérive structurelle incertaine** ; un horizon long ne l'améliore pas.

---

## 3. Ce que mesure notre propre banc

Source : `docs/BACKTEST_BASELINES_2026-09-29.md` (PR #99) et `scripts/stop_study.py` (résultats en annexe A). SXRV.DE, 2022-07-12 → 2026-09-29, capital 30 000 €, exécution à l'ouverture suivante, **coûts 25 pb par côté**, avant impôt.

| Stratégie | CAGR | Sharpe | Drawdown max | Trades |
|---|---|---|---|---|
| Buy & hold | 21,8 % | 1,12 | −26,7 % | 0 |
| MA200 avec hystérésis 2 % | 15,9 % | 1,06 | −15,0 % | 4 |
| Momentum 12 mois | 17,4 % | 1,06 | −33,0 % | 3 |
| Règles de sortie actuelles (entrée toujours haussière) | 8,6 % | 0,54 | −31,5 % | 83 |
| B&H + stop fixe −10 % depuis l'entrée | 21,2 % | 1,10 | −26,7 % | 1 |
| B&H + trailing −15 % depuis le sommet | 25,9 % | 1,36 | −24,3 % | 2 |
| B&H + trailing −25 % depuis le sommet | 17,5 % | 0,97 | −25,7 % | 1 |

Ce qu'on peut affirmer, et ce qu'on ne peut pas :

- **On peut affirmer** que la rotation des règles actuelles détruit de la valeur (ΔSharpe −0,58, IC 95 % [−0,89 ; −0,28]) et que, sur cette période, aucune règle simple ne bat le buy & hold sur le Sharpe.
- **On ne peut pas affirmer** que tel niveau de stop est « bon ». Le trailing −15 % paraît excellent (Sharpe 1,36) mais ses voisins −10 %, −20 %, −25 % ne le sont pas : avec seulement 1 à 5 événements par variante, c'est de la chance de paramètre. **Choisir ce niveau reviendrait à sur-ajuster.**
- **Sur le pétrole (proxy CL=F, non ajusté du roll)**, chaque variante de stop fait pire que le buy & hold ou à peu près pareil, et les stops suiveurs serrés multiplient les allers-retours (jusqu'à 22 trades) ; le pire drawdown (−65,5 %) vient du stop fixe −20 %, les stops suiveurs vont jusqu'à −64 %. Le proxy est peu fiable, mais le signe est cohérent avec le risque de « fouet ».
- **Limite du banc** : les stops y sont évalués sur les clôtures quotidiennes et exécutés à l'ouverture suivante, alors qu'un stop broker se déclenche en séance. Les événements de gap ne sont donc pas modélisés finement. Période : 4 ans, un seul cycle (baisse 2022, puis marché très haussier).

---

## 4. Options de conception

Trois questions distinctes, qu'il ne faut pas mélanger.

### 4.1 Vente : comment sort-on d'une position ?

| Option | Principe | Pour | Contre |
|---|---|---|---|
| **A. Jamais de vente automatique** | On détient l'indice, on ne vend que sur décision humaine | Aucun coût de rotation, aucun risque de fouet, cohérent avec le moyen/long terme, meilleur résultat de notre banc | Subit tout le drawdown (−26,7 % sur la période) ; dépend de ta capacité à tenir |
| **B. Stop de catastrophe très large** (−25/−30 % depuis le sommet, jamais remonté sous pression) | Assurance contre le scénario extrême | Se déclenche rarement (aucun déclenchement en stop fixe sur SXRV) ; limite une perte réellement destructrice | Gap possible ; ré-entrée à décider ; le niveau exact est arbitraire |
| **C. Sortie de régime lente** (clôture mensuelle sous MA200 avec hystérésis, ou momentum 12 mois < 0) | On sort quand la tendance de fond est cassée, décision hebdo ou mensuelle | Base académique la plus solide ; divise le drawdown par ~1,8 sur SXRV | Coûte du rendement (−6 pts/an en 2022-26) ; risque de fouet ; peu significatif statistiquement |
| **D. Réduction par la volatilité** | Exposition 100 % → 50 % quand la volatilité réalisée est haute | Pas de « tout ou rien » ; base académique (Moreira-Muir) | À tester sur nos séries ; plus complexe |
| **E. Combinaison** | Cœur A + garde-fou B ; C ou D en option, sur une fraction du capital | Séparer « ce que je veux garder » de « ce que je gère activement » | Complexité ; chaque brique doit prouver son apport |

### 4.2 Achat : comment entre-t-on ?

| Option | Principe | Remarque |
|---|---|---|
| **Une fois** | 100 % dès la décision | Gagne ~2/3 du temps contre le lissage, +1,2 à 2,4 pts de moyenne (Vanguard) |
| **Par paliers programmés** (3 à 12 mois) | Lissage mécanique | Réduit le regret et la variance d'entrée, coûte du rendement en moyenne ; cohérent si tu préfères dormir tranquille |
| **Filtré par le régime** | On entre par paliers seulement si le régime est haussier | Évite d'ouvrir en pleine cassure ; à backtester |

Point important : **le lot de 30 000 €** est un montant vécu comme « réel ». Le lissage est un choix de confort assumé, pas de performance.

### 4.3 Risque : par quoi le contrôle-t-on à moyen/long terme ?

1. **Par le choix de l'actif** : indice large plutôt que titres isolés (Bessembinder) ; éviter les ETC à roll pour la détention longue.
2. **Par la taille** : part du capital réellement investi, plafond par actif, réserve de cash ; réduction en régime de forte volatilité.
3. **Par le rythme de décision** : décision quotidienne ou hebdomadaire, pas toutes les 30 minutes (moins de bruit, moins de frais, moins de 429).
4. **Par un filet opérationnel**, distinct d'un stop de trading : alerte watchdog (déjà mergé), commande manuelle « tout liquider », arrêt sur perte de portefeuille définie à l'avance. Ce n'est pas un ordre de vente sur un niveau de prix de l'actif.

---

## 5. Proposition de travail (à valider par toi)

1. **Cœur** : indice actions large en EUR (SXRV.DE ou équivalent), **détenu durablement, sans stop de prix de trading**. Entrée : décision à prendre entre « en une fois » et « par paliers » (§4.2).
2. **Filet opérationnel** plutôt que stop de trading : watchdog, alertes, procédure de liquidation manuelle, plafond de perte portefeuille défini à l'avance. **Un stop broker très large (option B) reste une décision à part**, à prendre en connaissance du risque de gap.
3. **Surcouche de régime (option C ou D)** : optionnelle, sur une **fraction** du capital, jugée par le banc ; elle ne rentre que si elle bat le buy & hold et la MA200 simple hors échantillon, net de coûts et d'impôt.
4. **Pétrole** : sortir de la boucle de détention tant qu'il n'y a ni source de prix fiable ni instrument adapté (roll). Le laisser en satellite tactique éventuel, jugé à part.
5. **Modèles de l'ensemble (ML, TimesFM, LLM)** : ne pas les brancher sur la décision de vente/achat tant qu'ils n'ont pas prouvé leur apport dans le banc (tranche B). Ils peuvent alimenter le tableau de bord et l'analyse, sans droit de décision.

Ce que cela change dans le plan (#92) : §2.3 « stop catastrophe conservé » devient **une question ouverte** ; la phase 2 (règles de sortie) est **suspendue à ta décision** sur ce document.

---

## 6. Décisions à prendre

1. **Stop broker sur le cœur : oui / non / très large ?** (option A, B ou autre). Si non, l'invariant GO-gate 2 de `AGENTS.md` est à assouplir explicitement.
2. **Y a-t-il une surcouche active** (régime, volatilité), et sur quelle fraction du capital ?
3. **Entrée : en une fois ou par paliers**, et sur quelle durée ?
4. **Horizon et cadence** : décision quotidienne, hebdomadaire ou mensuelle ?
5. **Pétrole** : on l'abandonne, on le garde en satellite, ou on change d'instrument ?
6. **Ce que tu es prêt à supporter** : un drawdown de −27 % sur le cœur est-il acceptable en réel ? Cette réponse fixe plus de choses que n'importe quel paramètre.
7. **Fiscalité** : taux et régime à confirmer (hypothèse actuelle 30 % sur les plus-values réalisées) ; elle pénalise la rotation.

---

## 7. Comment trancher avec le banc (tranche B)

| Expérience | Question | Critère de décision |
|---|---|---|
| Cœur seul vs cœur + stop très large, plusieurs niveaux | Le stop de catastrophe a-t-il un coût ou un gain, et est-il robuste au niveau ? | Résultat stable sur ≥ 3 niveaux voisins, pas un optimum isolé |
| MA200 / momentum / volatilité sur **plusieurs actifs et périodes** (indices monde, Europe, obligations, or) | L'effet de régime existe-t-il hors du Nasdaq 2022-26 ? | Même signe sur la majorité des séries, IC du bootstrap ne contenant pas 0 |
| Entrée en une fois vs paliers | Combien coûte le lissage sur nos séries ? | Écart moyen et pire cas |
| Coût réel par côté | Le spread de SXRV.DE en compte réel | Mesure sur le compte réel avec de petits ordres, pas la démo |
| Instrument pétrole | Source fiable, roll, corrélation avec l'indice | Décision d'inclure ou non |

Règle d'or : **une variante n'est adoptée que si elle bat le buy & hold *et* la MA200 simple hors échantillon, net de coûts et d'impôt, avec une robustesse au paramètre.** Le Sharpe déflaté doit compter *toutes* les variantes essayées.

---

## Annexe A. Étude des stops (sortie brute de `scripts/stop_study.py`)

Stop évalué sur clôture quotidienne, exécution à l'ouverture suivante, ré-entrée après 21 séances si la base est toujours investie, coûts 25 pb par côté, avant impôt. Une ligne « sans stop » sert de référence.

### SXRV.DE (2022-07-12 → 2026-09-29)

| Variante | CAGR | Sharpe | Drawdown max | Trades | Temps investi |
|---|---|---|---|---|---|
| Buy & hold (sans stop) | 21,8 % | 1,12 | −26,7 % | 0 | 100 % |
| MA200 hystérésis (sans stop) | 15,9 % | 1,06 | −15,0 % | 4 | 81 % |
| B&H + stop fixe −15 % / −20 % / −25 % / −30 % depuis l'entrée | 21,8 % | 1,12 | −26,7 % | 0 | 100 % |
| B&H + stop fixe −10 % depuis l'entrée | 21,2 % | 1,10 | −26,7 % | 1 | 98 % |
| B&H + trailing −10 % depuis le sommet | 19,1 % | 1,06 | −20,7 % | 5 | 90 % |
| B&H + trailing −15 % depuis le sommet | 25,9 % | 1,36 | −24,3 % | 2 | 96 % |
| B&H + trailing −20 % depuis le sommet | 19,4 % | 1,07 | −23,0 % | 2 | 96 % |
| B&H + trailing −25 % depuis le sommet | 17,5 % | 0,97 | −25,7 % | 1 | 98 % |
| MA200 hystérésis + stop fixe −20 % | 13,4 % | 0,92 | −15,0 % | 4 | 79 % |

Le stop fixe −10 % se déclenche une fois et coûte 0,6 point de CAGR (21 séances hors marché avant la ré-entrée) ; au-delà, aucun déclenchement. L'état des stops (entrée, sommet) démarre à l'ouverture de la fenêtre testée ; seule la base MA200 utilise l'historique antérieur, pour être calculable dès le premier jour. Les stops fixes « inactifs » (0 trade) n'ont jamais été touchés : le drawdown de −26,7 % vient d'un sommet, pas du prix d'entrée du début de fenêtre.

### CL=F, proxy pétrole non ajusté du roll (2022-07-18 → 2026-09-29)

| Variante | CAGR | Sharpe | Drawdown max | Trades |
|---|---|---|---|---|
| Buy & hold (sans stop) | −2,1 % | 0,15 | −47,0 % | 0 |
| MA200 hystérésis (sans stop) | −8,7 % | −0,18 | −51,8 % | 9 |
| B&H + stop fixe −10 % depuis l'entrée | −7,2 % | −0,01 | −57,6 % | 6 |
| B&H + stop fixe −20 % depuis l'entrée | −11,6 % | −0,13 | −65,5 % | 4 |
| B&H + trailing −10 % depuis le sommet | −7,9 % | −0,14 | −56,1 % | 22 |
| B&H + trailing −15 % depuis le sommet | −11,7 % | −0,19 | −63,8 % | 13 |
| B&H + trailing −25 % depuis le sommet | −14,1 % | −0,24 | −61,0 % | 7 |

À lire avec la réserve du §3 : proxy non tradable, roll non ajusté.

## Sources

- [When Do Stop-Loss Rules Stop Losses? (Kaminski & Lo, SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338) ; [version Journal of Financial Markets](https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X) ; [MIT Open Access](https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf)
- [A Quantitative Approach to Tactical Asset Allocation (Faber)](https://www.cambriainvestments.com/wp-content/uploads/2018/01/A-Quantitative-Approach-to-Tactical-Asset-Allocation.pdf)
- [Time Series Momentum (Moskowitz, Ooi, Pedersen)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
- [A Century of Evidence on Trend-Following Investing (Hurst, Ooi, Pedersen)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026)
- [Momentum Crashes (Daniel & Moskowitz, NBER)](https://www.nber.org/papers/w20439)
- [Market Timing with Moving Averages: Anatomy and Performance of Trading Rules (Zakamulin, SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2585056) ; [Fooled by Data-Mining](https://www.researchgate.net/publication/256056022_Fooled_by_Data-Mining_The_Real-Life_Performance_of_Market_Timing_with_Moving_Average_and_Time-Series_Momentum_Rules)
- [Volatility-Managed Portfolios (Moreira & Muir, NBER)](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf)
- [Do Stocks Outperform Treasury Bills? (Bessembinder, SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2900447)
- [The truth about cost averaging (Vanguard)](https://www.nl.vanguard/professional/vanguard-365/cost-averaging)
- [The challenges of oil investing: Contango and the financialization of oil](https://www.sciencedirect.com/science/article/abs/pii/S0140988321003315) ; [Commodities for the Long Run (NBER)](https://www.nber.org/system/files/working_papers/w22793/w22793.pdf)
- [Trading 212 API : ordres](https://docs.trading212.com/api/orders/orders) ; [Trading 212 : ordres stop-limit](https://helpcentre.trading212.com/hc/en-us/articles/360007081297-Stop-Limit-Orders) ; [Trading 212 : types d'ordres](https://www.trading212.com/learn/investing-101/order-types-for-stocks)

## Décisions de l'utilisateur (2026-09-29)

| Question | Choix |
|---|---|
| Stop broker sur le cœur | **Aucun stop de prix.** Filet = alertes watchdog, liquidation manuelle, plafond de perte portefeuille fixé à l'avance. |
| Surcouche active (régime / volatilité) | **Oui, sur une fraction du capital** (ordre de grandeur 20 à 30 %), adoptée seulement si elle bat le buy & hold et la MA200 hors échantillon, net de coûts et d'impôt. |
| Entrée des 30 000 € | **En une fois.** |
| Pétrole | **Satellite tactique**, jugé à part ; source de prix fiable à trouver d'abord. |
| Cadence des décisions | **Hebdomadaire.** |
| Drawdown supportable sur le cœur | **Jusqu'à environ −30 %.** |

Conséquences : l'invariant GO-gate 2 (stop broker sur chaque position) doit être assoupli pour la stratégie cible, en le laissant en vigueur pour le démo actuel jusqu'à la refonte ; §2.3 du plan (#92) et la phase 2 sont à réécrire dans ce sens ; il reste à fixer la fraction exacte de la surcouche, à mesurer le spread réel et à choisir la source de prix du pétrole.
